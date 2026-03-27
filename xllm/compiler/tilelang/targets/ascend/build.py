from __future__ import annotations

import importlib
import os
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from ...common.cache import compute_cache_key, is_cache_hit
from ...common.manifest import (
    KernelAbi,
    KernelAbiParameter,
    KernelFamilyManifest,
    KernelVariantManifest,
)
from ...common.spec import (
    DispatchField,
    KernelCompileSpec,
    KernelSpec,
    TilelangKernel,
    is_registered_kernel_class,
)
from ...common.toolchain import (
    find_required_executable,
    git_head,
    prepare_tilelang_import,
    repo_root,
    require_env,
    run_checked,
)
from .kernels.utils import DEFAULT_ASCEND_BISHENG_ARCH
from .kernels.utils import render_family_registry_inc, render_family_variants_inc

TILELANG_BISHENG_COMMON_FLAGS = [
    "-O2",
    "-std=gnu++17",
    "-xcce",
    "-fPIC",
    "-mllvm",
    "-cce-aicore-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-function-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-record-overflow=true",
    "-mllvm",
    "-cce-aicore-addr-transform",
    "-mllvm",
    "-cce-aicore-dcci-insert-for-scalar=false",
    "-DL2_CACHE_HINT",
    "-DBACKEND_HYBM",
]

ASCEND_DEVICE_TO_BISHENG_ARCH = {
    "a2": DEFAULT_ASCEND_BISHENG_ARCH,
    "a3": DEFAULT_ASCEND_BISHENG_ARCH,
}


@dataclass(frozen=True)
class _RegisteredKernelFamily:
    module: ModuleType
    kernel_cls: type[TilelangKernel]
    module_name: str
    kernel_name: str
    dispatch_schema: list[DispatchField]
    spec_pairs: list[tuple[KernelCompileSpec, KernelSpec]]


def _load_kernel_module(module_name: str):
    prepare_tilelang_import()
    return importlib.import_module(f"{__package__}.kernels.{module_name}")


def _kernels_dir() -> Path:
    return Path(__file__).resolve().parent / "kernels"


def _iter_kernel_module_names() -> list[str]:
    return sorted(
        module.name
        for module in pkgutil.iter_modules([str(_kernels_dir())])
        if not module.name.startswith("_")
    )


def _resolve_registered_kernel_class(
    module_name: str,
) -> tuple[ModuleType, type[TilelangKernel] | None]:
    module = _load_kernel_module(module_name)
    kernel_classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and obj.__module__ == module.__name__
        and is_registered_kernel_class(obj)
    ]
    if not kernel_classes:
        return module, None
    if len(kernel_classes) > 1:
        kernel_names = ", ".join(sorted(cls.__name__ for cls in kernel_classes))
        raise TypeError(
            f"TileLang kernel module {module_name!r} must define at most one "
            f"@register_kernel class, found: {kernel_names}"
        )

    kernel_cls = kernel_classes[0]
    if not issubclass(kernel_cls, TilelangKernel):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must inherit "
            "TilelangKernel"
        )
    return module, kernel_cls


def _load_registered_kernel_family(
    module_name: str,
) -> _RegisteredKernelFamily | None:
    module, kernel_cls = _resolve_registered_kernel_class(module_name)
    if kernel_cls is None:
        return None

    generate_source = kernel_cls.__dict__.get("generate_source")
    if not isinstance(generate_source, (staticmethod, classmethod)):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must define "
            "callable generate_source(...) as @staticmethod or @classmethod"
        )

    resolved_generate_source = getattr(kernel_cls, "generate_source", None)
    if not callable(resolved_generate_source):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must define "
            "callable generate_source(...)"
        )

    resolved_specs = getattr(kernel_cls, "specs", None)
    if not callable(resolved_specs):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must define "
            "callable specs() -> list[KernelSpec]"
        )
    resolved_dispatch_schema = getattr(kernel_cls, "dispatch_schema", None)
    if not callable(resolved_dispatch_schema):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must define "
            "callable dispatch_schema() -> list[DispatchField]"
        )

    try:
        kernel_specs = resolved_specs()
    except NotImplementedError as exc:
        raise TypeError(str(exc)) from exc
    if not isinstance(kernel_specs, list) or not kernel_specs:
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must return a "
            "non-empty list[KernelSpec] from specs()"
        )
    try:
        dispatch_schema = resolved_dispatch_schema()
    except NotImplementedError as exc:
        raise TypeError(str(exc)) from exc
    if not isinstance(dispatch_schema, list) or not dispatch_schema:
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must return a "
            "non-empty list[DispatchField] from dispatch_schema()"
        )
    for index, field in enumerate(dispatch_schema):
        if not isinstance(field, DispatchField):
            raise TypeError(
                f"registered kernel class '{kernel_cls.__name__}' "
                f"dispatch_schema()[{index}] must be DispatchField"
            )

    family_kernel_name: str | None = None
    seen_variant_keys: set[str] = set()
    spec_pairs: list[tuple[KernelCompileSpec, KernelSpec]] = []

    for index, kernel_spec in enumerate(kernel_specs):
        if not isinstance(kernel_spec, KernelSpec):
            raise TypeError(
                f"registered kernel class '{kernel_cls.__name__}' specs()[{index}] "
                "must be KernelSpec"
            )

        kernel_spec.validate()
        missing_dispatch_fields = [
            field.name
            for field in dispatch_schema
            if field.name not in kernel_spec.specialization
        ]
        if missing_dispatch_fields:
            raise ValueError(
                f"registered kernel class '{kernel_cls.__name__}' specs()[{index}] "
                "is missing DISPATCH_SCHEMA fields: "
                f"{', '.join(missing_dispatch_fields)}"
            )
        compile_spec = kernel_spec.to_compile_spec(
            module_name=module_name,
            dispatch_schema=dispatch_schema,
        )
        if family_kernel_name is None:
            family_kernel_name = compile_spec.kernel_name
        elif compile_spec.kernel_name != family_kernel_name:
            raise ValueError(
                f"registered kernel class '{kernel_cls.__name__}' must return "
                "KernelSpec entries with the same kernel_name"
            )

        if compile_spec.variant_key in seen_variant_keys:
            raise ValueError(
                f"registered kernel class '{kernel_cls.__name__}' has duplicate "
                f"variant_key {compile_spec.variant_key!r}"
            )
        seen_variant_keys.add(compile_spec.variant_key)
        spec_pairs.append((compile_spec, kernel_spec))

    assert family_kernel_name is not None
    return _RegisteredKernelFamily(
        module=module,
        kernel_cls=kernel_cls,
        module_name=module_name,
        kernel_name=family_kernel_name,
        dispatch_schema=dispatch_schema,
        spec_pairs=spec_pairs,
    )


def _registered_families() -> dict[str, _RegisteredKernelFamily]:
    families: dict[str, _RegisteredKernelFamily] = {}
    for module_name in _iter_kernel_module_names():
        family = _load_registered_kernel_family(module_name)
        if family is None:
            continue
        if family.kernel_name in families:
            raise ValueError(
                "Duplicate Ascend TileLang kernel_name registered: "
                f"{family.kernel_name}"
            )
        families[family.kernel_name] = family
    return families


def get_default_families(
    kernel_names: list[str] | None = None,
) -> list[_RegisteredKernelFamily]:
    families = _registered_families()
    if kernel_names is None:
        return list(families.values())
    missing = [name for name in kernel_names if name not in families]
    if missing:
        raise ValueError(f"Unknown Ascend TileLang kernels: {', '.join(missing)}")
    return [families[name] for name in kernel_names]


def _rename_entry_symbol(source: str, source_entry_symbol: str, entry_symbol: str) -> str:
    pattern = rf"\b{re.escape(source_entry_symbol)}\b"
    return re.sub(pattern, entry_symbol, source)


def _rename_variant_internal_symbols(source: str, variant_key: str) -> str:
    symbol_names: set[str] = set()
    symbol_names.update(
        re.findall(
            r'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
            source,
        )
    )
    symbol_names.update(
        re.findall(r"\bvoid\s+([A-Za-z_][A-Za-z0-9_]*_tiling)\s*\(", source)
    )

    renamed_source = source
    for symbol_name in sorted(symbol_names, key=len, reverse=True):
        renamed_source = re.sub(
            rf"\b{re.escape(symbol_name)}\b",
            f"{symbol_name}__{variant_key}",
            renamed_source,
        )
    return renamed_source


def _normalize_cpp_type(cpp_type: str) -> str:
    normalized = re.sub(r"\s+", " ", cpp_type).strip()
    normalized = re.sub(r"\s*([*&]+)\s*", r"\1", normalized)
    return normalized


def _parse_kernel_abi(source: str, entry_symbol: str) -> KernelAbi:
    pattern = re.compile(
        rf'extern\s+"C"\s+'
        rf"(?P<return_type>[^(){{}};]+?)\s+"
        rf"{re.escape(entry_symbol)}\s*\("
        r"(?P<params>[^)]*)\)\s*\{",
        re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        raise ValueError(
            f"Failed to parse exported entry ABI for symbol {entry_symbol!r}"
        )

    return_type = _normalize_cpp_type(match.group("return_type"))
    params_text = match.group("params").strip()
    parameters: list[KernelAbiParameter] = []
    if params_text and params_text != "void":
        for param in (part.strip() for part in params_text.split(",")):
            parsed = re.match(
                r"(?P<type>.+?[\*&]?)\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)$",
                param,
            )
            if parsed is None:
                raise ValueError(
                    "Failed to parse kernel ABI parameter "
                    f"{param!r} for symbol {entry_symbol!r}"
                )
            parameters.append(
                KernelAbiParameter(
                    cpp_type=_normalize_cpp_type(parsed.group("type")),
                    name=parsed.group("name"),
                )
            )

    return KernelAbi(return_type=return_type, parameters=parameters)


def _build_fingerprint(bisheng_executable: str, bisheng_arch: str) -> dict[str, str]:
    tl_root = require_env("TL_ROOT")
    npu_home_path = _resolve_npu_home_path()
    return {
        "target": "ascend",
        "tl_root": tl_root,
        "tilelang_git_head": git_head(tl_root),
        "npu_home_path": npu_home_path,
        "bisheng_executable": bisheng_executable,
        "bisheng_arch": bisheng_arch,
    }


def _normalize_ascend_device(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = device.strip().lower()
    if not normalized:
        return None
    if normalized not in ASCEND_DEVICE_TO_BISHENG_ARCH:
        supported = ", ".join(sorted(ASCEND_DEVICE_TO_BISHENG_ARCH))
        raise ValueError(
            f"Unsupported Ascend TileLang device {device!r}. Expected one of: "
            f"{supported}"
        )
    return normalized


def _resolve_bisheng_arch(device: str | None) -> tuple[str | None, str]:
    normalized_device = _normalize_ascend_device(device)
    if normalized_device is None:
        print(
            "[WARN] TileLang Ascend build did not receive --device. Falling back "
            f"to default bisheng_arch={DEFAULT_ASCEND_BISHENG_ARCH}. Prefer "
            "running via xLLM main build path or pass --device a2|a3 explicitly."
        )
        return None, DEFAULT_ASCEND_BISHENG_ARCH
    return normalized_device, ASCEND_DEVICE_TO_BISHENG_ARCH[normalized_device]


def _build_toolchain_options(device: str | None, bisheng_arch: str) -> dict[str, str]:
    toolchain_options = {"bisheng_arch": bisheng_arch}
    if device is not None:
        toolchain_options["device"] = device
    return toolchain_options


def _resolve_npu_home_path() -> str:
    for env_name in ("NPU_HOME_PATH", "NPU_TOOLKIT_HOME"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    for candidate in (
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend-toolkit",
    ):
        if Path(candidate).exists():
            return candidate

    raise RuntimeError(
        "Required NPU toolkit root is not set. Expected NPU_HOME_PATH or "
        "NPU_TOOLKIT_HOME, or a standard install path under "
        "/usr/local/Ascend/ascend-toolkit."
    )


def _bisheng_include_dirs() -> list[str]:
    tl_root = require_env("TL_ROOT")
    npu_home_path = _resolve_npu_home_path()
    return [
        f"{npu_home_path}/include",
        f"{npu_home_path}/include/experiment/runtime",
        f"{npu_home_path}/include/experiment/msprof",
        f"{npu_home_path}/compiler/tikcpp",
        f"{npu_home_path}/compiler/tikcpp/tikcfw",
        f"{npu_home_path}/compiler/tikcpp/tikcfw/impl",
        f"{npu_home_path}/compiler/tikcpp/tikcfw/interface",
        f"{tl_root}/3rdparty/catlass/include",
        f"{tl_root}/3rdparty/shmem/include",
        f"{tl_root}/3rdparty/shmem/src/device",
        f"{tl_root}/src",
    ]


def _variant_entry_symbol(spec: KernelCompileSpec) -> str:
    kernel_entry_name = spec.entry_name or spec.kernel_name
    return f"{kernel_entry_name}__{spec.variant_key}_call"


def _read_family_manifest(path: Path) -> KernelFamilyManifest | None:
    if not path.is_file():
        return None
    try:
        return KernelFamilyManifest.read(path)
    except Exception:
        return None


def _render_variants_inc(
    kernel_name: str,
    kernel_cls: type[TilelangKernel],
    dispatch_schema: list[DispatchField],
    variants: list[KernelVariantManifest],
) -> str:
    renderer = getattr(kernel_cls, "render_variants_inc", None)
    if renderer is None:
        return render_family_variants_inc(
            kernel_name=kernel_name,
            dispatch_schema=dispatch_schema,
            variants=variants,
        )
    if not callable(renderer):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' defines "
            "non-callable render_variants_inc"
        )
    rendered = renderer(variants, dispatch_schema)
    if not isinstance(rendered, str):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' "
            "render_variants_inc(...) must return str"
        )
    return rendered


def _render_registry_inc(
    kernel_name: str,
    kernel_cls: type[TilelangKernel],
    dispatch_schema: list[DispatchField],
    kernel_abi: KernelAbi,
    variants: list[KernelVariantManifest],
) -> str:
    renderer = getattr(kernel_cls, "render_registry_inc", None)
    if renderer is None:
        return render_family_registry_inc(
            kernel_name=kernel_name,
            dispatch_schema=dispatch_schema,
            kernel_abi=kernel_abi,
            variants=variants,
        )
    if not callable(renderer):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' defines "
            "non-callable render_registry_inc"
        )
    rendered = renderer(variants, dispatch_schema, kernel_abi)
    if not isinstance(rendered, str):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' "
            "render_registry_inc(...) must return str"
        )
    return rendered


def build_kernel_family(
    family: _RegisteredKernelFamily,
    output_root: str | Path,
    force: bool = False,
    device: str | None = None,
    bisheng_arch: str = DEFAULT_ASCEND_BISHENG_ARCH,
) -> KernelFamilyManifest:
    family_output_dir = Path(output_root) / "targets" / "ascend" / family.kernel_name
    family_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = family_output_dir / "manifest.json"
    existing_manifest = _read_family_manifest(manifest_path)

    bisheng_executable = find_required_executable("bisheng")
    variant_manifests: list[KernelVariantManifest] = []
    family_kernel_abi: KernelAbi | None = None
    toolchain_options = _build_toolchain_options(device=device, bisheng_arch=bisheng_arch)

    for compile_spec, kernel_spec in family.spec_pairs:
        if compile_spec.target != "ascend":
            raise ValueError(
                f"Unsupported target for Ascend build.py: {compile_spec.target}"
            )

        variant_output_dir = family_output_dir / compile_spec.variant_key
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        generated_source = (
            variant_output_dir
            / f"{compile_spec.kernel_name}_{compile_spec.variant_key}_kernel.cpp"
        )
        compiled_binary = (
            variant_output_dir
            / f"{compile_spec.kernel_name}_{compile_spec.variant_key}_kernel.o"
        )

        fingerprint = _build_fingerprint(bisheng_executable, bisheng_arch)
        if device is not None:
            fingerprint["device"] = device
        dependency_files = [
            Path(family.module.__file__).resolve(),
            Path(__file__).resolve(),
        ]
        cache_key = compute_cache_key(compile_spec, fingerprint, dependency_files)

        cached_variant = (
            existing_manifest.get_variant(compile_spec.variant_key)
            if existing_manifest is not None
            else None
        )
        if (
            not force
            and cached_variant is not None
            and Path(cached_variant.generated_source).is_file()
            and Path(cached_variant.compiled_binary).is_file()
            and is_cache_hit(manifest_path, compile_spec.variant_key, cache_key)
        ):
            cached_source = Path(cached_variant.generated_source).read_text(
                encoding="utf-8"
            )
            kernel_abi = _parse_kernel_abi(cached_source, cached_variant.entry_symbol)
            if family_kernel_abi is None:
                family_kernel_abi = kernel_abi
            elif kernel_abi != family_kernel_abi:
                raise ValueError(
                    "All variants in a TileLang kernel must share the same exported "
                    f"C ABI. Mismatch found in variant {compile_spec.variant_key!r}."
                )
            variant_manifests.append(
                KernelVariantManifest(
                    variant_key=compile_spec.variant_key,
                    specialization=dict(compile_spec.specialization),
                    dispatch_values=dict(compile_spec.dispatch_values),
                    generated_source=cached_variant.generated_source,
                    compiled_binary=cached_variant.compiled_binary,
                    entry_symbol=cached_variant.entry_symbol,
                    cache_key=cached_variant.cache_key,
                    toolchain_options=dict(toolchain_options),
                    fingerprint=dict(fingerprint),
                    compile_definitions=kernel_spec.render_compile_definitions(
                        entry_symbol=cached_variant.entry_symbol
                    ),
                )
            )
            continue

        source = family.kernel_cls.generate_source(**compile_spec.specialization)
        entry_symbol = _variant_entry_symbol(compile_spec)
        rendered_source = _rename_variant_internal_symbols(
            _rename_entry_symbol(
                source, compile_spec.source_entry_symbol, entry_symbol
            ),
            compile_spec.variant_key,
        )
        kernel_abi = _parse_kernel_abi(rendered_source, entry_symbol)
        if family_kernel_abi is None:
            family_kernel_abi = kernel_abi
        elif kernel_abi != family_kernel_abi:
            raise ValueError(
                "All variants in a TileLang kernel must share the same exported "
                f"C ABI. Mismatch found in variant {compile_spec.variant_key!r}."
            )
        generated_source.write_text(rendered_source, encoding="utf-8")

        compile_cmd = [
            bisheng_executable,
            f"--cce-aicore-arch={bisheng_arch}",
            *TILELANG_BISHENG_COMMON_FLAGS,
            f"-Dg_tilingKey=g_tilingKey__{compile_spec.variant_key}",
            *[f"-I{include_dir}" for include_dir in _bisheng_include_dirs()],
            str(generated_source),
            "-c",
            "-o",
            str(compiled_binary),
        ]
        run_checked(compile_cmd, cwd=repo_root())

        variant_manifests.append(
            KernelVariantManifest(
                variant_key=compile_spec.variant_key,
                specialization=compile_spec.specialization,
                dispatch_values=compile_spec.dispatch_values,
                generated_source=str(generated_source),
                compiled_binary=str(compiled_binary),
                entry_symbol=entry_symbol,
                cache_key=cache_key,
                toolchain_options=dict(toolchain_options),
                fingerprint=fingerprint,
                compile_definitions=kernel_spec.render_compile_definitions(
                    entry_symbol=entry_symbol
                ),
            )
        )

    if family_kernel_abi is None:
        raise ValueError(
            f"TileLang kernel {family.kernel_name!r} produced no exported kernel ABI"
        )

    variants_inc_path = family_output_dir / "variants.inc"
    variants_inc_path.write_text(
        _render_variants_inc(
            family.kernel_name,
            family.kernel_cls,
            family.dispatch_schema,
            variant_manifests,
        ),
        encoding="utf-8",
    )
    registry_inc_path = family_output_dir / "registry.inc"
    registry_inc_path.write_text(
        _render_registry_inc(
            family.kernel_name,
            family.kernel_cls,
            family.dispatch_schema,
            family_kernel_abi,
            variant_manifests,
        ),
        encoding="utf-8",
    )

    manifest = KernelFamilyManifest(
        target="ascend",
        kernel_name=family.kernel_name,
        output_dir=str(family_output_dir),
        variants_inc=str(variants_inc_path),
        registry_inc=str(registry_inc_path),
        dispatch_schema=list(family.dispatch_schema),
        kernel_abi=family_kernel_abi,
        variants=variant_manifests,
    )
    manifest.write(manifest_path)
    return manifest


def build_kernels(
    output_root: str | Path,
    kernel_names: list[str] | None = None,
    force: bool = False,
    device: str | None = None,
) -> list[KernelFamilyManifest]:
    normalized_device, bisheng_arch = _resolve_bisheng_arch(device)
    manifests = []
    for family in get_default_families(kernel_names):
        manifests.append(
            build_kernel_family(
                family,
                output_root=output_root,
                force=force,
                device=normalized_device,
                bisheng_arch=bisheng_arch,
            )
        )
    return manifests
