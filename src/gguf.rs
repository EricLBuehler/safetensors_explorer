#![allow(unused, non_camel_case_types)]

use anyhow::Result;
use std::collections::HashMap;
use std::io::{Cursor, Read};

/// GGUF file format parser
/// Based on llama.cpp GGUF specification
pub struct GGUFFile {
    pub header: GGUFHeader,
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: Vec<GGUFTensorInfo>,
}

#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GGMLType,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
}

/// GGML tensor types from llama.cpp
/// Includes all quantization formats
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    GGML_TYPE_Q1_58 = 36,
}

impl GGMLType {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GGMLType::F32),
            1 => Some(GGMLType::F16),
            2 => Some(GGMLType::Q4_0),
            3 => Some(GGMLType::Q4_1),
            6 => Some(GGMLType::Q5_0),
            7 => Some(GGMLType::Q5_1),
            8 => Some(GGMLType::Q8_0),
            9 => Some(GGMLType::Q8_1),
            10 => Some(GGMLType::Q2_K),
            11 => Some(GGMLType::Q3_K),
            12 => Some(GGMLType::Q4_K),
            13 => Some(GGMLType::Q5_K),
            14 => Some(GGMLType::Q6_K),
            15 => Some(GGMLType::Q8_K),
            16 => Some(GGMLType::IQ2_XXS),
            17 => Some(GGMLType::IQ2_XS),
            18 => Some(GGMLType::IQ3_XXS),
            19 => Some(GGMLType::IQ1_S),
            20 => Some(GGMLType::IQ4_NL),
            21 => Some(GGMLType::IQ3_S),
            22 => Some(GGMLType::IQ2_S),
            23 => Some(GGMLType::IQ4_XS),
            24 => Some(GGMLType::I8),
            25 => Some(GGMLType::I16),
            26 => Some(GGMLType::I32),
            27 => Some(GGMLType::I64),
            28 => Some(GGMLType::F64),
            29 => Some(GGMLType::IQ1_M),
            30 => Some(GGMLType::BF16),
            36 => Some(GGMLType::GGML_TYPE_Q1_58),
            _ => None,
        }
    }

    /// Get the size in bytes per element for this type
    /// For quantized types, this is an approximation
    pub fn element_size_bytes(&self) -> f32 {
        match self {
            GGMLType::F32 | GGMLType::I32 => 4.0,
            GGMLType::F16 | GGMLType::BF16 | GGMLType::I16 => 2.0,
            GGMLType::F64 | GGMLType::I64 => 8.0,
            GGMLType::I8 => 1.0,

            // Legacy Q‑quants (block of 32 weights)
            GGMLType::Q4_0 => 0.5625, // 18  / 32  bytes
            GGMLType::Q4_1 => 0.625,  // 20  / 32
            GGMLType::Q5_0 => 0.6875, // 22  / 32
            GGMLType::Q5_1 => 0.75,   // 24  / 32
            GGMLType::Q8_0 => 1.0625, // 34  / 32
            GGMLType::Q8_1 => 1.125,  // 36  / 32

            // K‑quants (super‑block of 256 weights)
            GGMLType::Q2_K => 0.328_125,   // 2.625  bpw
            GGMLType::Q3_K => 0.429_687_5, // 3.4375 bpw
            GGMLType::Q4_K => 0.5625,      // 4.5    bpw
            GGMLType::Q5_K => 0.6875,      // 5.5    bpw
            GGMLType::Q6_K => 0.820_312_5, // 6.5625 bpw
            GGMLType::Q8_K => 1.140_625,   // 9.125  bpw

            // Importance‑quants (IQ‑family, super‑block 256)
            GGMLType::IQ1_S => 0.195_312_5,   // 1.5625 bpw
            GGMLType::IQ1_M => 0.218_75,      // 1.75   bpw
            GGMLType::IQ2_XXS => 0.257_812_5, // 2.0625 bpw
            GGMLType::IQ2_XS => 0.289_062_5,  // 2.3125 bpw
            GGMLType::IQ2_S => 0.3125,        // 2.5    bpw
            GGMLType::IQ3_XXS => 0.382_812_5, // 3.0625 bpw
            GGMLType::IQ3_S => 0.429_687_5,   // 3.4375 bpw
            GGMLType::IQ4_NL => 0.53125,      // 4.25   bpw
            GGMLType::IQ4_XS => 0.53125,      // 4.25   bpw
            GGMLType::GGML_TYPE_Q1_58 => 0.1975, // 1.58 / 8
        }
    }
}

impl std::fmt::Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            GGMLType::F32 => "F32",
            GGMLType::F16 => "F16",
            GGMLType::F64 => "F64",
            GGMLType::BF16 => "BF16",
            GGMLType::I8 => "I8",
            GGMLType::I16 => "I16",
            GGMLType::I32 => "I32",
            GGMLType::I64 => "I64",
            GGMLType::Q4_0 => "Q4_0",
            GGMLType::Q4_1 => "Q4_1",
            GGMLType::Q5_0 => "Q5_0",
            GGMLType::Q5_1 => "Q5_1",
            GGMLType::Q8_0 => "Q8_0",
            GGMLType::Q8_1 => "Q8_1",
            GGMLType::Q2_K => "Q2_K",
            GGMLType::Q3_K => "Q3_K",
            GGMLType::Q4_K => "Q4_K",
            GGMLType::Q5_K => "Q5_K",
            GGMLType::Q6_K => "Q6_K",
            GGMLType::Q8_K => "Q8_K",
            GGMLType::IQ2_XXS => "IQ2_XXS",
            GGMLType::IQ2_XS => "IQ2_XS",
            GGMLType::IQ3_XXS => "IQ3_XXS",
            GGMLType::IQ1_S => "IQ1_S",
            GGMLType::IQ4_NL => "IQ4_NL",
            GGMLType::IQ3_S => "IQ3_S",
            GGMLType::IQ2_S => "IQ2_S",
            GGMLType::IQ4_XS => "IQ4_XS",
            GGMLType::IQ1_M => "IQ1_M",
            GGMLType::GGML_TYPE_Q1_58 => "Q1_58",
        };
        write!(f, "{s}")
    }
}

impl std::fmt::Display for GGUFValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGUFValue::U8(v) => write!(f, "{v}"),
            GGUFValue::I8(v) => write!(f, "{v}"),
            GGUFValue::U16(v) => write!(f, "{v}"),
            GGUFValue::I16(v) => write!(f, "{v}"),
            GGUFValue::U32(v) => write!(f, "{v}"),
            GGUFValue::I32(v) => write!(f, "{v}"),
            GGUFValue::F32(v) => write!(f, "{v}"),
            GGUFValue::U64(v) => write!(f, "{v}"),
            GGUFValue::I64(v) => write!(f, "{v}"),
            GGUFValue::F64(v) => write!(f, "{v}"),
            GGUFValue::Bool(v) => write!(f, "{v}"),
            GGUFValue::String(v) => write!(f, "\"{v}\""),
            GGUFValue::Array(arr) => {
                if arr.len() <= 5 {
                    // Show small arrays in full
                    write!(f, "[")?;
                    for (i, item) in arr.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{item}")?;
                    }
                    write!(f, "]")
                } else {
                    // Show truncated arrays
                    write!(
                        f,
                        "[{}, {}, ..., {} ({})]",
                        arr[0],
                        arr[1],
                        arr[arr.len() - 1],
                        arr.len()
                    )
                }
            }
        }
    }
}

impl GGUFFile {
    pub fn read(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // Read header
        let header = Self::read_header(&mut cursor)?;

        // Validate magic number
        if header.magic != 0x46554747 {
            return Err(anyhow::anyhow!("Invalid GGUF magic number"));
        }

        // Read metadata
        let metadata = Self::read_metadata(&mut cursor, header.metadata_kv_count)?;

        // Read tensor info
        let tensors = Self::read_tensor_info(&mut cursor, header.tensor_count)?;

        Ok(GGUFFile {
            header,
            metadata,
            tensors,
        })
    }

    fn read_header(cursor: &mut Cursor<&[u8]>) -> Result<GGUFHeader> {
        let magic = Self::read_u32(cursor)?;
        let version = Self::read_u32(cursor)?;
        let tensor_count = Self::read_u64(cursor)?;
        let metadata_kv_count = Self::read_u64(cursor)?;

        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    fn read_metadata(cursor: &mut Cursor<&[u8]>, count: u64) -> Result<HashMap<String, GGUFValue>> {
        let mut metadata = HashMap::new();

        for _ in 0..count {
            let key = Self::read_string(cursor)?;
            let value_type = Self::read_u32(cursor)?;
            let value = Self::read_value(cursor, value_type)?;
            metadata.insert(key, value);
        }

        Ok(metadata)
    }

    fn read_tensor_info(cursor: &mut Cursor<&[u8]>, count: u64) -> Result<Vec<GGUFTensorInfo>> {
        let mut tensors = Vec::new();

        for _ in 0..count {
            let name = Self::read_string(cursor)?;
            let n_dimensions = Self::read_u32(cursor)?;
            let mut dimensions = Vec::new();

            for _ in 0..n_dimensions {
                dimensions.push(Self::read_u64(cursor)?);
            }

            let tensor_type_u32 = Self::read_u32(cursor)?;
            let tensor_type = GGMLType::from_u32(tensor_type_u32)
                .ok_or_else(|| anyhow::anyhow!("Unknown tensor type: {}", tensor_type_u32))?;

            let offset = Self::read_u64(cursor)?;

            tensors.push(GGUFTensorInfo {
                name,
                dimensions,
                tensor_type,
                offset,
            });
        }

        Ok(tensors)
    }

    fn read_value(cursor: &mut Cursor<&[u8]>, value_type: u32) -> Result<GGUFValue> {
        match value_type {
            0 => Ok(GGUFValue::U8(Self::read_u8(cursor)?)),
            1 => Ok(GGUFValue::I8(Self::read_i8(cursor)?)),
            2 => Ok(GGUFValue::U16(Self::read_u16(cursor)?)),
            3 => Ok(GGUFValue::I16(Self::read_i16(cursor)?)),
            4 => Ok(GGUFValue::U32(Self::read_u32(cursor)?)),
            5 => Ok(GGUFValue::I32(Self::read_i32(cursor)?)),
            6 => Ok(GGUFValue::F32(Self::read_f32(cursor)?)),
            7 => Ok(GGUFValue::Bool(Self::read_u8(cursor)? != 0)),
            8 => Ok(GGUFValue::String(Self::read_string(cursor)?)),
            9 => {
                let array_type = Self::read_u32(cursor)?;
                let array_len = Self::read_u64(cursor)?;
                let mut array = Vec::new();
                for _ in 0..array_len {
                    array.push(Self::read_value(cursor, array_type)?);
                }
                Ok(GGUFValue::Array(array))
            }
            10 => Ok(GGUFValue::U64(Self::read_u64(cursor)?)),
            11 => Ok(GGUFValue::I64(Self::read_i64(cursor)?)),
            12 => Ok(GGUFValue::F64(Self::read_f64(cursor)?)),
            _ => Err(anyhow::anyhow!("Unknown value type: {}", value_type)),
        }
    }

    fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
        let len = Self::read_u64(cursor)?;
        let mut bytes = vec![0u8; len as usize];
        cursor.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    }

    fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
        let mut buf = [0u8; 1];
        cursor.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_i8(cursor: &mut Cursor<&[u8]>) -> Result<i8> {
        Ok(Self::read_u8(cursor)? as i8)
    }

    fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
        let mut buf = [0u8; 2];
        cursor.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(cursor: &mut Cursor<&[u8]>) -> Result<i16> {
        let mut buf = [0u8; 2];
        cursor.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
        let mut buf = [0u8; 4];
        cursor.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
        let mut buf = [0u8; 4];
        cursor.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
        let mut buf = [0u8; 4];
        cursor.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
        let mut buf = [0u8; 8];
        cursor.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(cursor: &mut Cursor<&[u8]>) -> Result<i64> {
        let mut buf = [0u8; 8];
        cursor.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f64(cursor: &mut Cursor<&[u8]>) -> Result<f64> {
        let mut buf = [0u8; 8];
        cursor.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }
}
