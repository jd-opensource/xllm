// copied from https://github.com/mlc-ai/tokenizers-cpp/blob/v0.1.1/rust/src/lib.rs

// A simple C wrapper of tokenzier library
use std::{collections::HashMap, str::FromStr};
use std::fs;
use std::ffi::{c_char, CStr};
use std::io;
use tokenizers::tokenizer::Tokenizer;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    decode_str: String,
    id_to_token_result: String,
}

pub type Vocab = HashMap<String, u32>;
pub type Merges = Vec<(String, String)>;

#[repr(C)]
pub struct TokenizerEncodeResult {
    token_ids: *mut u32,
    len: usize,
}

fn read_file_as_u8(path: &str) -> Result<Vec<u8>, io::Error> {
    fs::read(path)
}

impl TokenizerWrapper {
    pub fn from_str(json: &str) -> TokenizerWrapper {
        TokenizerWrapper {
            tokenizer: Tokenizer::from_str(json).unwrap().into(),
            decode_str: String::new(),
            id_to_token_result: String::new(),
        }
    }

    pub fn encode(&mut self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let encoded = self.tokenizer.encode(text, add_special_tokens).unwrap();
        return encoded.get_ids().to_vec();
    }

    pub fn encode_batch(&mut self, texts: Vec<&str>, add_special_tokens: bool) -> Vec<Vec<u32>> {
        let results = self.tokenizer.encode_batch(texts, add_special_tokens).unwrap()
            .into_iter()
            .map(|encoded| encoded.get_ids().to_vec())
            .collect::<Vec<Vec<u32>>>();
        return results;
    }

    pub fn decode(&mut self, ids: &[u32], skip_special_tokens: bool) {
        self.decode_str = self.tokenizer.decode(ids, skip_special_tokens).unwrap();
    }
}

#[no_mangle]
extern "C" fn tokenizers_new_from_str(input_cstr: *const u8, len: usize) -> *mut TokenizerWrapper {
    unsafe {
        let json = &String::from_utf8_lossy(std::slice::from_raw_parts(input_cstr, len));
        return Box::into_raw(Box::new(TokenizerWrapper::from_str(json)));
    }
}

#[no_mangle]
extern "C" fn tokenizers_new_from_path(path: *const c_char) -> *mut TokenizerWrapper {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => panic!("Failed to convert C path string to Rust string"),
    };

    match read_file_as_u8(path_str) {
        Ok(bytes) => {
            return tokenizers_new_from_str(bytes.as_ptr(), bytes.len());
        }
        Err(_) => {
            panic!("Failed to read tokenizer file.");
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_encode(
    handle: *mut TokenizerWrapper,
    input_cstr: *const u8,
    len: usize,
    add_special_tokens: i32,
    out_result: *mut TokenizerEncodeResult,
) {
    unsafe {
        let input_data = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        let encoded = (*handle).encode(input_data, add_special_tokens != 0);
        let len = encoded.len();
        *out_result = TokenizerEncodeResult {
            token_ids: Box::into_raw(encoded.into_boxed_slice()) as *mut u32,
            len: len,
        };
    }
}

#[no_mangle]
extern "C" fn tokenizers_encode_batch(
    handle: *mut TokenizerWrapper,
    input_cstr: *const *const u8,
    input_len: *const usize,
    num_seqs: usize,
    add_special_tokens: i32,
    out_result: *mut TokenizerEncodeResult,
) {
    unsafe {
        let input_data = (0..num_seqs)
            .map(|i| {
                std::str::from_utf8(std::slice::from_raw_parts(*input_cstr.offset(i as isize), *input_len.offset(i as isize))).unwrap()
            })
            .collect::<Vec<&str>>();
        let encoded_batch = (*handle).encode_batch(input_data, add_special_tokens != 0);
        for (i, encoded) in encoded_batch.into_iter().enumerate() {
            let len = encoded.len();
            let result = TokenizerEncodeResult {
                token_ids: Box::into_raw(encoded.into_boxed_slice()) as *mut u32,
                len: len,
            };
            *out_result.offset(i as isize) = result;
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_free_encode_results(results: *mut TokenizerEncodeResult, num_seqs: usize) {
    unsafe {
        let slice = std::slice::from_raw_parts_mut(results, num_seqs);
        for result in &mut *slice {
            drop(Box::from_raw(std::slice::from_raw_parts_mut(result.token_ids, result.len)));
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_decode(
    handle: *mut TokenizerWrapper,
    input_ids: *const u32,
    len: usize,
    skip_special_tokens: i32,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        let input_data = std::slice::from_raw_parts(input_ids, len);
        (*handle).decode(input_data, skip_special_tokens != 0);

        *out_cstr = (*handle).decode_str.as_mut_ptr();
        *out_len = (&(*handle).decode_str).len();
    }
}

#[no_mangle]
extern "C" fn tokenizers_free(wrapper: *mut TokenizerWrapper) {
    unsafe {
        drop(Box::from_raw(wrapper));
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_vocab_size(handle: *mut TokenizerWrapper, size: *mut usize) {
    unsafe {
        *size = (*handle).tokenizer.get_vocab_size(true);
    }
}

#[no_mangle]
extern "C" fn tokenizers_id_to_token(
    handle: *mut TokenizerWrapper,
    id: u32,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        let str = (*handle).tokenizer.id_to_token(id);
        (*handle).id_to_token_result = match str {
            Some(s) => s,
            None => String::from(""),
        };

        *out_cstr = (*handle).id_to_token_result.as_mut_ptr();
        *out_len = (&(*handle).id_to_token_result).len();
    }
}

#[no_mangle]
extern "C" fn tokenizers_token_to_id(
    handle: *mut TokenizerWrapper,
    token: *const u8,
    len: usize,
    out_id: *mut i32,
) {
    unsafe {
        let token: &str = &String::from_utf8_lossy(std::slice::from_raw_parts(token, len));
        let id = (*handle).tokenizer.token_to_id(token);
        *out_id = match id {
            Some(id) => id as i32,
            None => -1,
        };
    }
}
