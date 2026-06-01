// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include <string>

#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace ck_tile {
/*
 * a host side utility, arg parser for, either
 * -[key0] = [value0, value1, value2]
 * or
 * -[key0]=[value0] -[key1]=[value1] ...
 */
class ArgParser
{
    public:
    class Arg
    {
        public:
        std::string name;
        std::string value;
        std::string help_text;
    };

    ArgParser() {}
    ArgParser& insert(const std::string& _name,
                      const std::string& _default_value,
                      const std::string& _help_text)
    {
        Arg in;
        in.name      = _name;
        in.value     = _default_value;
        in.help_text = _help_text;

        if(input_map.count(_name) != 0)
        {
            printf("arg:%s already exist\n", _name.c_str());
        }
        else
        {
            input_map[_name] = in;
            keys.push_back(_name);
        }
        return *this;
    }
    void print() const
    {
        // find max key length
        std::string::size_type max_key_length = 11;
        for(auto& key : keys)
        {
            if(max_key_length < key.length())
            {
                max_key_length = key.length();
            }
        }

        printf("args:\n");
        for(auto& key : keys)
        {
            auto value = input_map.at(key);
            std::vector<std::string> help_text_lines;
            size_t pos = 0;
            for(size_t next_pos = value.help_text.find('\n', pos); next_pos != std::string::npos;)
            {
                help_text_lines.push_back(std::string(value.help_text.begin() + pos,
                                                      value.help_text.begin() + next_pos++));
                pos      = next_pos;
                next_pos = value.help_text.find('\n', pos);
            }
            help_text_lines.push_back(
                std::string(value.help_text.begin() + pos, value.help_text.end()));

            std::string default_value = std::string("(default:") + value.value + std::string(")");
            std::cout << std::setw(1 + max_key_length - value.name.length()) << "-" << key
                      << std::setw(4) << " " << help_text_lines[0] << " " << default_value
                      << std::endl;

            for(auto help_next_line = std::next(help_text_lines.begin());
                help_next_line != help_text_lines.end();
                ++help_next_line)
            {
                std::cout << std::setw(1 + max_key_length + 4) << " " << *help_next_line
                          << std::endl;
            }
        }
    }
    bool parse(int argc, char* argv[], int start_index = 1)
    {
        if(argc < start_index)
        {
            printf("not enough args\n");
            return false;
        }
        for(int i = start_index; i < argc; i++)
        {
            char* cur_arg = argv[i];
            if(cur_arg[0] != '-')
            {
                printf("illegal input\n");
                print();
                return false;
            }
            else
            {
                std::string text(cur_arg + 1);
                if(text == "?")
                {
                    print();
                    return false;
                }
                auto pos = text.find('=');
                if(pos == std::string::npos)
                {
                    printf("arg should be [key]=[value] pair, here:%s\n", text.c_str());
                    return false;
                }
                if(pos >= (text.size() - 1))
                {
                    printf("cant find value after \"=\", here:%s\n", text.c_str());
                    return false;
                }
                auto key   = text.substr(0, pos);
                auto value = text.substr(pos + 1);
                if(input_map.count(key) == 0)
                {
                    printf("no such arg:%s\n", key.c_str());
                    return false;
                }
                input_map[key].value = value;
            }
        }
        return true;
    }

    std::string get_str(const std::string& name) const
    {
        std::string value = input_map.at(name).value;
        return value;
    }

    int get_int(const std::string& name) const
    {
        int value = atoi(input_map.at(name).value.c_str());
        return value;
    }

    uint32_t get_uint32(const std::string& name) const
    {
        uint32_t value = strtoul(input_map.at(name).value.c_str(), nullptr, 10);
        return value;
    }

    uint64_t get_uint64(const std::string& name) const
    {
        uint64_t value = strtoull(input_map.at(name).value.c_str(), nullptr, 10);
        return value;
    }

    bool get_bool(const std::string& name) const
    {
        auto v = input_map.at(name).value;
        if(v.compare("t") == 0 || v.compare("true") == 0)
            return true;
        if(v.compare("f") == 0 || v.compare("false") == 0)
            return false;
        int value = atoi(v.c_str());
        return value == 0 ? false : true;
    }

    float get_float(const std::string& name) const
    {
        double value = atof(input_map.at(name).value.c_str());
        return static_cast<float>(value);
    }

    double get_double(const std::string& name) const
    {
        double value = atof(input_map.at(name).value.c_str());
        return value;
    }

    std::vector<std::string> get_string_vec(const std::string& name,
                                            const std::string& delimiter = ",") const
    {
        if(get_str(name).empty())
        {
            return {};
        }
        std::string s = get_str(name);
        std::vector<std::string> tokens;
        size_t pos = 0;
        std::string token;
        while((pos = s.find(delimiter)) != std::string::npos)
        {
            token = s.substr(0, pos);
            tokens.push_back(token);
            s.erase(0, pos + delimiter.length());
        }
        tokens.push_back(s);

        return tokens;
    }

    std::vector<int> get_int_vec(const std::string& name, const std::string& delimiter = ",") const
    {
        if(get_str(name).empty())
        {
            return {};
        }
        const std::vector<std::string> args = get_string_vec(name, delimiter);
        std::vector<int> tokens;
        tokens.reserve(static_cast<int>(args.size()));
        for(const std::string& token : args)
        {
            int value = atoi(token.c_str());
            tokens.push_back(value);
        }
        return tokens;
    }

    private:
    std::unordered_map<std::string, Arg> input_map;
    std::vector<std::string> keys;
};
} // namespace ck_tile
