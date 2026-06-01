// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

namespace ck_tile {

template <typename... Args>
void CK_TILE_ERROR(Args&&... args) noexcept
{
    std::ostringstream oss;
    (oss << ... << args);
    std::cerr << "[ERROR] " << oss.str() << std::endl;
}

namespace internal {

template <size_t N>
bool is_any_of(const char* const (&names)[N], const std::string& str)
{
    return std::any_of(std::begin(names), std::end(names), [&](const char* inner_str) {
        return str == inner_str;
    });
};

template <typename T>
struct ParseEnvVal
{
};
template <>
struct ParseEnvVal<bool>
{
    static bool parse_env_var_value(const char* vp)
    {
        std::string value_env_str{vp};

        for(auto& c : value_env_str)
        {
            if(std::isalpha(c) != 0)
            {
                c = std::tolower(static_cast<unsigned char>(c));
            }
        }

        if(is_any_of(enabled_names, value_env_str))
        {
            return true;
        }
        else if(is_any_of(disabled_names, value_env_str))
        {
            return false;
        }
        else
        {
            throw std::runtime_error("Invalid value for env variable");
        }

        return false;
    }

    private:
    static constexpr const char* enabled_names[]  = {"enable", "enabled", "1", "yes", "on", "true"};
    static constexpr const char* disabled_names[] = {
        "disable", "disabled", "0", "no", "off", "false"};
};

// Supports hexadecimals (with leading "0x"), octals (if prefix is "0") and decimals (default).
// Returns 0 if environment variable is in wrong format (strtoull fails to parse the string).
template <>
struct ParseEnvVal<uint64_t>
{
    static uint64_t parse_env_var_value(const char* vp) { return std::strtoull(vp, nullptr, 0); }
};

template <>
struct ParseEnvVal<std::string>
{
    static std::string parse_env_var_value(const char* vp) { return std::string{vp}; }
};

template <typename T>
struct EnvVar
{
    private:
    T value{};
    bool is_unset = true;

    public:
    const T& GetValue() const { return value; }

    bool IsUnset() const { return is_unset; }

    void Unset() { is_unset = true; }

    void UpdateValue(const T& val)
    {
        is_unset = false;
        value    = val;
    }

    explicit EnvVar(const char* const name, const T& def_val)
    {
        // NOLINTNEXTLINE (concurrency-mt-unsafe)
        const char* vp = std::getenv(name);
        if(vp != nullptr) // a value was provided
        {
            is_unset = false;
            value    = ParseEnvVal<T>::parse_env_var_value(vp);
        }
        else // no value provided, use default value
        {
            value = def_val;
        }
    }
};
} // end namespace internal

// Static inside function hides the variable and provides
// thread-safety/locking
// Used in global namespace
#define CK_TILE_DECLARE_ENV_VAR(name, type, default_val)                            \
    namespace ck_tile::env {                                                        \
    struct name                                                                     \
    {                                                                               \
        static_assert(std::is_same_v<name, ::ck_tile::env::name>,                   \
                      "CK_TILE_DECLARE_ENV* must be used in the global namespace"); \
        using value_type = type;                                                    \
        static ck_tile::internal::EnvVar<type>& Ref()                               \
        {                                                                           \
            static ck_tile::internal::EnvVar<type> var{#name, default_val};         \
            return var;                                                             \
        }                                                                           \
    };                                                                              \
    }

#define CK_TILE_DECLARE_ENV_VAR_BOOL(name) CK_TILE_DECLARE_ENV_VAR(name, bool, false)

#define CK_TILE_DECLARE_ENV_VAR_UINT64(name) CK_TILE_DECLARE_ENV_VAR(name, uint64_t, 0)

#define CK_TILE_DECLARE_ENV_VAR_STR(name) CK_TILE_DECLARE_ENV_VAR(name, std::string, "")

#define CK_TILE_ENV(name) \
    ck_tile::env::name {}

template <class EnvVar>
inline const std::string& EnvGetString(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, std::string>);
    return EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool EnvIsEnabled(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, bool>);
    return !EnvVar::Ref().IsUnset() && EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool EnvIsDisabled(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, bool>);
    return !EnvVar::Ref().IsUnset() && !EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline uint64_t EnvValue(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, uint64_t>);
    return EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool EnvIsUnset(EnvVar)
{
    return EnvVar::Ref().IsUnset();
}

template <class EnvVar>
void EnvUnset(EnvVar)
{
    EnvVar::Ref().Unset();
}

/// Updates the cached value of an environment variable
template <typename EnvVar, typename ValueType>
void UpdateEnvVar(EnvVar, const ValueType& val)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, ValueType>);
    EnvVar::Ref().UpdateValue(val);
}

template <typename EnvVar>
void UpdateEnvVar(EnvVar, const std::string_view& val)
{
    EnvVar::Ref().UpdateValue(
        ck_tile::internal::ParseEnvVal<typename EnvVar::value_type>::parse_env_var_value(
            val.data()));
}

} // namespace ck_tile

// environment variable to enable logging:
// export CK_TILE_LOGGING=ON or CK_TILE_LOGGING=1 or CK_TILE_LOGGING=ENABLED
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_LOGGING)
