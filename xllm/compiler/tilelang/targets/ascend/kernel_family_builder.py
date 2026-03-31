from __future__ import annotations

from pathlib import Path

from ...common.cache import compute_cache_key, is_cache_hit
from ...common.manifest import KernelAbi, KernelFamilyManifest, KernelVariantManifest
from ...common.spec import DispatchField, KernelCompileSpec, TilelangKernel
from ...common.toolchain import repo_root, run_checked
from . import abi_entry, kernel_registry, toolchain
from .kernel_registry import RegisteredKernelFamily
from .kernels import utils as kernel_utils
from .kernels.utils import render_family_registry_inc, render_family_variants_inc
from .toolchain import AscendBuildContext, TILELANG_BISHENG_COMMON_FLAGS


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


def _build_dependency_files(family: RegisteredKernelFamily) -> list[Path]:
    # Keep cache invalidation aligned with the split builder implementation.
    files = [
        Path(family.module.__file__).resolve(),
        Path(__file__).resolve(),
        Path(toolchain.__file__).resolve(),
        Path(kernel_registry.__file__).resolve(),
        Path(abi_entry.__file__).resolve(),
        Path(kernel_utils.__file__).resolve(),
        Path(__file__).resolve().with_name("build.py"),
    ]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def build_kernel_family(
    family: RegisteredKernelFamily,
    output_root: str | Path,
    context: AscendBuildContext,
    force: bool = False,
) -> KernelFamilyManifest:
    family_output_dir = Path(output_root) / "targets" / "ascend" / family.kernel_name
    family_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = family_output_dir / "manifest.json"
    existing_manifest = _read_family_manifest(manifest_path)
    dependency_files = _build_dependency_files(family)

    variant_manifests: list[KernelVariantManifest] = []
    family_kernel_abi: KernelAbi | None = None

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

        cache_key = compute_cache_key(
            compile_spec,
            context.fingerprint,
            dependency_files,
        )

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
            kernel_abi = abi_entry.parse_kernel_abi(
                cached_source, cached_variant.entry_symbol
            )
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
                    toolchain_options=dict(context.toolchain_options),
                    fingerprint=dict(context.fingerprint),
                    compile_definitions=kernel_spec.render_compile_definitions(
                        entry_symbol=cached_variant.entry_symbol
                    ),
                )
            )
            continue

        source = family.kernel_cls.generate_source(**compile_spec.specialization)
        entry_symbol = _variant_entry_symbol(compile_spec)
        rendered_source = abi_entry.rename_variant_internal_symbols(
            abi_entry.rename_entry_symbol(
                source, compile_spec.source_entry_symbol, entry_symbol
            ),
            compile_spec.variant_key,
        )
        kernel_abi = abi_entry.parse_kernel_abi(rendered_source, entry_symbol)
        if family_kernel_abi is None:
            family_kernel_abi = kernel_abi
        elif kernel_abi != family_kernel_abi:
            raise ValueError(
                "All variants in a TileLang kernel must share the same exported "
                f"C ABI. Mismatch found in variant {compile_spec.variant_key!r}."
            )
        generated_source.write_text(rendered_source, encoding="utf-8")

        compile_cmd = [
            context.bisheng_executable,
            f"--cce-aicore-arch={context.bisheng_arch}",
            *TILELANG_BISHENG_COMMON_FLAGS,
            f"-Dg_tilingKey=g_tilingKey__{compile_spec.variant_key}",
            *[f"-I{include_dir}" for include_dir in context.include_dirs],
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
                toolchain_options=dict(context.toolchain_options),
                fingerprint=dict(context.fingerprint),
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
