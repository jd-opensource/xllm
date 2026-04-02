/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <gflags/gflags.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"

namespace xllm {

class HelpFormatter {
 public:
  static std::string generate_help() {
    std::ostringstream oss;
    const auto sections = build_help_sections();
    const auto model_flag = get_required_model_flag();

    size_t public_flag_count = 0;
    for (const auto& section : sections) {
      public_flag_count += section.flags.size();
    }

    oss << "USAGE: xllm --model <PATH> [OPTIONS]\n\n";
    oss << "xLLM help is generated from the registered global flags.\n";
    oss << "Section order comes from xllm/core/common/global_flags.cpp, and "
           "type/default/description come from gflags metadata.\n";
    oss << "Field order is stable so both humans and agents can parse it "
           "reliably.\n\n";

    oss << "REQUIRED OPTIONS:\n";
    append_option(oss, model_flag);

    oss << "HELP OPTIONS:\n";
    append_virtual_option(
        oss,
        /*option_name=*/"-h, --help",
        /*option_type=*/"bool",
        /*default_value=*/"false",
        /*description=*/"Display this help message and exit.");

    oss << "AVAILABLE FLAG SECTIONS: " << public_flag_count
        << " public xLLM gflags across " << sections.size() << " sections\n\n";

    for (const auto& section : sections) {
      oss << section.metadata.title << ":\n";
      if (!section.metadata.summary.empty()) {
        append_wrapped_text(oss,
                            section.metadata.summary,
                            /*first_indent=*/"  ",
                            /*continuation_indent=*/"  ");
        oss << "\n";
      }
      for (const auto& option_info : section.flags) {
        append_option(oss, option_info);
      }
    }

    oss << "For more information and all available options, visit:\n";
    oss << "  https://github.com/jd-opensource/xllm/blob/main/xllm/core/common/"
           "global_flags.cpp\n";
    oss << "Documentation: "
           "https://xllm.readthedocs.io/zh-cn/latest/cli_reference/\n";

    return oss.str();
  }

  static void print_help() { std::cout << generate_help(); }

  static void print_usage() {
    std::cout << "USAGE: xllm --model <PATH> [OPTIONS]\n";
    std::cout << "Try 'xllm --help' for a structured view of all public xLLM "
                 "flags.\n";
  }

  static void print_error(const std::string& error_msg) {
    std::cerr << "Error: " << error_msg << "\n\n";
    print_usage();
  }

 private:
  struct HelpSectionView final {
    GlobalFlagHelpSectionInfo metadata;
    std::vector<google::CommandLineFlagInfo> flags;
  };

  static bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
               0;
  }

  static bool is_xllm_global_flag(
      const google::CommandLineFlagInfo& option_info) {
    return ends_with(option_info.filename, "global_flags.cpp");
  }

  static std::string default_value_for_display(
      const google::CommandLineFlagInfo& option_info) {
    if (option_info.type == "string" && option_info.default_value.empty()) {
      return "\"\"";
    }
    return option_info.default_value;
  }

  static void append_wrapped_text(std::ostringstream& oss,
                                  const std::string& text,
                                  const std::string& first_indent,
                                  const std::string& continuation_indent,
                                  size_t width = 100) {
    if (text.empty()) {
      return;
    }

    std::istringstream words(text);
    std::string word;
    size_t current_width = first_indent.size();
    bool first_word = true;

    oss << first_indent;
    while (words >> word) {
      const size_t extra_width = first_word ? 0 : 1;
      if (!first_word && current_width + extra_width + word.size() > width) {
        oss << "\n" << continuation_indent;
        current_width = continuation_indent.size();
        first_word = true;
      }

      if (!first_word) {
        oss << ' ';
        ++current_width;
      }

      oss << word;
      current_width += word.size();
      first_word = false;
    }
    oss << "\n";
  }

  static void append_field(std::ostringstream& oss,
                           const std::string& name,
                           const std::string& value) {
    const std::string first_indent = "      " + name + ": ";
    const std::string continuation_indent(first_indent.size(), ' ');
    append_wrapped_text(oss,
                        value,
                        /*first_indent=*/first_indent,
                        /*continuation_indent=*/continuation_indent);
  }

  static std::unordered_map<std::string, google::CommandLineFlagInfo>
  collect_public_flag_map() {
    std::vector<google::CommandLineFlagInfo> all_flags;
    google::GetAllFlags(&all_flags);

    std::unordered_map<std::string, google::CommandLineFlagInfo> public_flags;
    public_flags.reserve(all_flags.size());
    for (const auto& option_info : all_flags) {
      if (!is_xllm_global_flag(option_info)) {
        continue;
      }
      public_flags.emplace(option_info.name, option_info);
    }
    return public_flags;
  }

  static std::vector<HelpSectionView> build_help_sections() {
    const auto public_flags = collect_public_flag_map();

    std::vector<GlobalFlagHelpSectionInfo> section_order(
        get_global_flag_help_sections().begin(),
        get_global_flag_help_sections().end());
    std::sort(section_order.begin(),
              section_order.end(),
              [](const GlobalFlagHelpSectionInfo& lhs,
                 const GlobalFlagHelpSectionInfo& rhs) {
                return lhs.order < rhs.order;
              });

    std::vector<GlobalFlagHelpFlagInfo> flag_order(
        get_global_flag_help_flags().begin(),
        get_global_flag_help_flags().end());
    std::sort(flag_order.begin(),
              flag_order.end(),
              [](const GlobalFlagHelpFlagInfo& lhs,
                 const GlobalFlagHelpFlagInfo& rhs) {
                return lhs.order < rhs.order;
              });

    std::vector<HelpSectionView> sections;
    sections.reserve(section_order.size() + 1);
    std::unordered_map<std::string, size_t> section_index_by_key;
    section_index_by_key.reserve(section_order.size());
    for (size_t index = 0; index < section_order.size(); ++index) {
      sections.emplace_back(HelpSectionView{section_order[index], {}});
      section_index_by_key.emplace(section_order[index].key, index);
    }

    std::unordered_set<std::string> consumed_flags;
    consumed_flags.reserve(public_flags.size());
    for (const auto& help_flag : flag_order) {
      if (help_flag.name == "model") {
        continue;
      }

      const auto flag_it = public_flags.find(help_flag.name);
      if (flag_it == public_flags.end()) {
        continue;
      }

      const auto section_it = section_index_by_key.find(help_flag.section_key);
      if (section_it == section_index_by_key.end()) {
        continue;
      }

      sections[section_it->second].flags.emplace_back(flag_it->second);
      consumed_flags.insert(help_flag.name);
    }

    std::vector<google::CommandLineFlagInfo> uncategorized_flags;
    for (const auto& [name, option_info] : public_flags) {
      if (name == "model" || consumed_flags.contains(name)) {
        continue;
      }
      uncategorized_flags.emplace_back(option_info);
    }
    std::sort(uncategorized_flags.begin(),
              uncategorized_flags.end(),
              [](const google::CommandLineFlagInfo& lhs,
                 const google::CommandLineFlagInfo& rhs) {
                return lhs.name < rhs.name;
              });

    if (!uncategorized_flags.empty()) {
      sections.emplace_back(
          HelpSectionView{{"uncategorized_options",
                           "UNCATEGORIZED OPTIONS",
                           "Flags registered in global_flags.cpp "
                           "without an explicit help section.",
                           0},
                          std::move(uncategorized_flags)});
    }

    sections.erase(std::remove_if(sections.begin(),
                                  sections.end(),
                                  [](const HelpSectionView& section) {
                                    return section.flags.empty();
                                  }),
                   sections.end());
    return sections;
  }

  static void append_option(std::ostringstream& oss,
                            const google::CommandLineFlagInfo& option_info) {
    oss << "  --" << option_info.name << "\n";
    append_field(oss, /*name=*/"type", /*value=*/option_info.type);
    append_field(oss,
                 /*name=*/"default",
                 /*value=*/default_value_for_display(option_info));
    append_field(oss,
                 /*name=*/"description",
                 /*value=*/option_info.description.empty()
                     ? "No description provided."
                     : option_info.description);
    oss << "\n";
  }

  static void append_virtual_option(std::ostringstream& oss,
                                    const std::string& option_name,
                                    const std::string& option_type,
                                    const std::string& default_value,
                                    const std::string& description) {
    oss << "  " << option_name << "\n";
    append_field(oss, /*name=*/"type", /*value=*/option_type);
    append_field(oss, /*name=*/"default", /*value=*/default_value);
    append_field(oss, /*name=*/"description", /*value=*/description);
    oss << "\n";
  }

  static google::CommandLineFlagInfo get_required_model_flag() {
    google::CommandLineFlagInfo option_info;
    google::GetCommandLineFlagInfo("model", &option_info);
    return option_info;
  }
};

}  // namespace xllm
