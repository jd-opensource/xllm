#ifndef PARTIAL_JSON_PARSER_OPTIONS_H
#define PARTIAL_JSON_PARSER_OPTIONS_H

namespace partial_json_parser {

enum TypeOptions {
  STR = 1 << 0,        // 1
  NUM = 1 << 1,        // 2
  ARR = 1 << 2,        // 4
  OBJ = 1 << 3,        // 8
  NULL_TYPE = 1 << 4,  // 16 (using NULL_TYPE to avoid conflict with NULL macro)
  BOOL = 1 << 5,       // 32
  NAN_TYPE = 1 << 6,   // 64 (using NAN_TYPE to avoid conflict with NAN macro)
  INFINITY_TYPE =
      1
      << 7,  // 128 (using INFINITY_TYPE to avoid conflict with INFINITY macro)
  NEG_INFINITY = 1 << 8,  // 256

  INF = INFINITY_TYPE | NEG_INFINITY,
  SPECIAL = NULL_TYPE | BOOL | INF | NAN_TYPE,
  ATOM = STR | NUM | SPECIAL,
  COLLECTION = ARR | OBJ,
  ALL = ATOM | COLLECTION
};

}  // namespace partial_json_parser

#endif  // PARTIAL_JSON_PARSER_OPTIONS_H