use crate::{
    boxtree::{
        types::{
            BrickData, MIPMapStrategy, MIPResamplingMethods, NodeChildren, NodeContent, NodeData,
            PaletteIndexValues,
        },
        Albedo, BoxTree, BOX_NODE_CHILDREN_COUNT,
    },
    object_pool::ObjectPool,
    Version,
};
use bendy::{
    decoding::{Error, FromBencode, Object},
    encoding::{Error as BencodeError, SingleItemEncoder, ToBencode},
};
use std::{collections::HashMap, fmt::Debug, hash::Hash};

impl ToBencode for Version {
    const MAX_DEPTH: usize = 2;
    fn encode(&self, encoder: SingleItemEncoder) -> Result<(), BencodeError> {
        encoder.emit_list(|e| {
            e.emit_int(self.major())?;
            e.emit_int(self.minor())?;
            e.emit_int(self.patch())
        })
    }
}

impl FromBencode for Version {
    fn decode_bencode_object(data: Object) -> Result<Self, bendy::decoding::Error> {
        match data {
            Object::List(mut list) => {
                let major = match list.next_object()?.expect("Expected Major version string") {
                    Object::Integer(i) => Ok(i.parse::<u32>()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field library major version",
                        "Something else",
                    )),
                }?;
                let minor = match list.next_object()?.expect("Expected Minor version string") {
                    Object::Integer(i) => Ok(i.parse::<u32>()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field library major version",
                        "Something else",
                    )),
                }?;
                let patch = match list.next_object()?.expect("Expected Patch version string") {
                    Object::Integer(i) => Ok(i.parse::<u32>()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field library major version",
                        "Something else",
                    )),
                }?;
                Ok(crate::Version {
                    major,
                    minor,
                    patch,
                })
            }
            _ => Err(bendy::decoding::Error::unexpected_token("List", "not List")),
        }
    }
}

//####################################################################################
//  █████   █████    ███████    █████ █████ ██████████ █████
// ░░███   ░░███   ███░░░░░███ ░░███ ░░███ ░░███░░░░░█░░███
//  ░███    ░███  ███     ░░███ ░░███ ███   ░███  █ ░  ░███
//  ░███    ░███ ░███      ░███  ░░█████    ░██████    ░███
//  ░░███   ███  ░███      ░███   ███░███   ░███░░█    ░███
//   ░░░█████░   ░░███     ███   ███ ░░███  ░███ ░   █ ░███      █
//     ░░███      ░░░███████░   █████ █████ ██████████ ███████████
//      ░░░         ░░░░░░░    ░░░░░ ░░░░░ ░░░░░░░░░░ ░░░░░░░░░░░
//    █████████     ███████    ██████   █████ ███████████ ██████████ ██████   █████ ███████████
//   ███░░░░░███  ███░░░░░███ ░░██████ ░░███ ░█░░░███░░░█░░███░░░░░█░░██████ ░░███ ░█░░░███░░░█
//  ███     ░░░  ███     ░░███ ░███░███ ░███ ░   ░███  ░  ░███  █ ░  ░███░███ ░███ ░   ░███  ░
// ░███         ░███      ░███ ░███░░███░███     ░███     ░██████    ░███░░███░███     ░███
// ░███         ░███      ░███ ░███ ░░██████     ░███     ░███░░█    ░███ ░░██████     ░███
// ░░███     ███░░███     ███  ░███  ░░█████     ░███     ░███ ░   █ ░███  ░░█████     ░███
//  ░░█████████  ░░░███████░   █████  ░░█████    █████    ██████████ █████  ░░█████    █████
//   ░░░░░░░░░     ░░░░░░░    ░░░░░    ░░░░░    ░░░░░    ░░░░░░░░░░ ░░░░░    ░░░░░    ░░░░░
//####################################################################################
impl ToBencode for Albedo {
    const MAX_DEPTH: usize = 2;
    fn encode(&self, encoder: SingleItemEncoder) -> Result<(), BencodeError> {
        encoder.emit_list(|e| {
            e.emit(self.r)?;
            e.emit(self.g)?;
            e.emit(self.b)?;
            e.emit(self.a)
        })
    }
}

impl FromBencode for Albedo {
    fn decode_bencode_object(data: Object) -> Result<Self, bendy::decoding::Error> {
        match data {
            Object::List(mut list) => {
                let r = match list.next_object()?.unwrap() {
                    Object::Integer(i) => Ok(i.parse()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field red color component",
                        "Something else",
                    )),
                }?;
                let g = match list.next_object()?.unwrap() {
                    Object::Integer(i) => Ok(i.parse()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field green color component",
                        "Something else",
                    )),
                }?;
                let b = match list.next_object()?.unwrap() {
                    Object::Integer(i) => Ok(i.parse()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field blue color component",
                        "Something else",
                    )),
                }?;
                let a = match list.next_object()?.unwrap() {
                    Object::Integer(i) => Ok(i.parse()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field alpha color component",
                        "Something else",
                    )),
                }?;
                Ok(Self { r, g, b, a })
            }
            _ => Err(bendy::decoding::Error::unexpected_token("List", "not List")),
        }
    }
}

//####################################################################################
//  ███████████  ███████████   █████   █████████  █████   ████
// ░░███░░░░░███░░███░░░░░███ ░░███   ███░░░░░███░░███   ███░
//  ░███    ░███ ░███    ░███  ░███  ███     ░░░  ░███  ███
//  ░██████████  ░██████████   ░███ ░███          ░███████
//  ░███░░░░░███ ░███░░░░░███  ░███ ░███          ░███░░███
//  ░███    ░███ ░███    ░███  ░███ ░░███     ███ ░███ ░░███
//  ███████████  █████   █████ █████ ░░█████████  █████ ░░████
// ░░░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░░░░░  ░░░░░   ░░░░
//  ██████████     █████████   ███████████   █████████
// ░░███░░░░███   ███░░░░░███ ░█░░░███░░░█  ███░░░░░███
//  ░███   ░░███ ░███    ░███ ░   ░███  ░  ░███    ░███
//  ░███    ░███ ░███████████     ░███     ░███████████
//  ░███    ░███ ░███░░░░░███     ░███     ░███░░░░░███
//  ░███    ███  ░███    ░███     ░███     ░███    ░███
//  ██████████   █████   █████    █████    █████   █████
//####################################################################################
impl<T> ToBencode for BrickData<T>
where
    T: ToBencode + Default + Clone + PartialEq,
{
    const MAX_DEPTH: usize = 3;

    fn encode(&self, encoder: SingleItemEncoder) -> Result<(), BencodeError> {
        match self {
            BrickData::Empty => encoder.emit_str("#b"),
            BrickData::Solid(voxel) => encoder.emit_list(|e| {
                e.emit_str("#b#")?;
                e.emit(voxel)
            }),
            BrickData::Parted(brick) => encoder.emit_list(|e| {
                e.emit_str("##b#")?;
                e.emit_int(brick.len())?;
                for voxel in brick.iter() {
                    e.emit(voxel)?;
                }
                e.emit_str("#")?;
                Ok(())
            }),
        }
    }
}

impl<T> FromBencode for BrickData<T>
where
    T: FromBencode + Clone + PartialEq,
{
    fn decode_bencode_object(data: Object) -> Result<Self, bendy::decoding::Error> {
        match data {
            Object::Bytes(b) => {
                debug_assert_eq!(
                    String::from_utf8(b.to_vec())
                        .unwrap_or("".to_string())
                        .as_str(),
                    "#b"
                );
                Ok(BrickData::Empty)
            }
            Object::List(mut list) => {
                let is_solid = match list.next_object()?.unwrap() {
                    Object::Bytes(b) => {
                        match String::from_utf8(b.to_vec())
                            .unwrap_or("".to_string())
                            .as_str()
                        {
                            "#b#" => Ok(true),   // The content is a single voxel
                            "##b#" => Ok(false), // The content is a brick of voxels
                            misc => Err(bendy::decoding::Error::unexpected_token(
                                "A NodeContent Identifier string, which is either # or ##",
                                "The string ".to_owned() + misc,
                            )),
                        }
                    }
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "BrickData string identifier",
                        "Something else",
                    )),
                }?;
                if is_solid {
                    Ok(BrickData::Solid(T::decode_bencode_object(
                        list.next_object()?.unwrap(),
                    )?))
                } else {
                    let len = match list.next_object()?.unwrap() {
                        Object::Integer(i) => Ok(i.parse()?),
                        _ => Err(bendy::decoding::Error::unexpected_token(
                            "int field brick length",
                            "Something else",
                        )),
                    }?;
                    debug_assert!(0 < len, "Expected brick to be of non-zero length!");
                    let mut brick_data = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        brick_data.push(T::decode_bencode_object(list.next_object()?.unwrap())?);
                    }
                    Ok(BrickData::Parted(brick_data))
                }
            }
            _ => Err(bendy::decoding::Error::unexpected_token(
                "A NodeContent Object, either a List or a ByteString",
                "Something else",
            )),
        }
    }
}

//####################################################################################
//  ██████   █████    ███████    ██████████   ██████████
// ░░██████ ░░███   ███░░░░░███ ░░███░░░░███ ░░███░░░░░█
//  ░███░███ ░███  ███     ░░███ ░███   ░░███ ░███  █ ░
//  ░███░░███░███ ░███      ░███ ░███    ░███ ░██████
//  ░███ ░░██████ ░███      ░███ ░███    ░███ ░███░░█
//  ░███  ░░█████ ░░███     ███  ░███    ███  ░███ ░   █
//  █████  ░░█████ ░░░███████░   ██████████   ██████████
// ░░░░░    ░░░░░    ░░░░░░░    ░░░░░░░░░░   ░░░░░░░░░░
//  ██████████     █████████   ███████████   █████████
// ░░███░░░░███   ███░░░░░███ ░█░░░███░░░█  ███░░░░░███
//  ░███   ░░███ ░███    ░███ ░   ░███  ░  ░███    ░███
//  ░███    ░███ ░███████████     ░███     ░███████████
//  ░███    ░███ ░███░░░░░███     ░███     ░███░░░░░███
//  ░███    ███  ░███    ░███     ░███     ░███    ░███
//  ██████████   █████   █████    █████    █████   █████
//####################################################################################
impl ToBencode for NodeData {
    const MAX_DEPTH: usize = SERIALIZE_MAX_DEPTH;
    fn encode(&self, encoder: SingleItemEncoder) -> Result<(), BencodeError> {
        encoder.emit_list(|e| {
            e.emit(&self.content)?;
            e.emit(self.children)?;
            e.emit(&self.mip)?;
            e.emit(self.occupied_bits)?;
            e.emit(self.occlusion_bits)
        })
    }
}

impl FromBencode for NodeData {
    fn decode_bencode_object(data: Object) -> Result<Self, bendy::decoding::Error> {
        match data {
            Object::List(mut list) => {
                let content = NodeContent::<u32>::decode_bencode_object(
                    list.next_object()?
                        .expect("Expected Node children from byte stream!"),
                )?;
                let children = NodeChildren::decode_bencode_object(
                    list.next_object()?
                        .expect("Expected Node contents from byte stream!"),
                )?;
                let mip = BrickData::<PaletteIndexValues>::decode_bencode_object(
                    list.next_object()?
                        .expect("Expected Node mip from byte stream!"),
                )?;
                let occupied_bits = u64::decode_bencode_object(
                    list.next_object()?
                        .expect("Expected Node occupied bits from byte stream!"),
                )?;
                let occlusion_bits = u8::decode_bencode_object(
                    list.next_object()?
                        .expect("Expected Node occlusion bits from byte stream!"),
                )?;
                Ok(Self {
                    content,
                    children,
                    mip,
                    occupied_bits,
                    occlusion_bits,
                })
            }
            _ => Err(bendy::decoding::Error::unexpected_token("List", "not List")),
        }
    }
}

impl<T> ToBencode for NodeContent<T>
where
    T: ToBencode + Debug + Default + Clone + PartialEq,
{
    const MAX_DEPTH: usize = 8;
    fn encode(&self, encoder: SingleItemEncoder) -> Result<(), BencodeError> {
        match self {
            NodeContent::Nothing => encoder.emit_str("#"),
            NodeContent::Internal => encoder.emit_str("##"),
            NodeContent::Leaf(bricks) => encoder.emit_list(|e| {
                e.emit_str("###")?;
                for brick in bricks.iter().take(BOX_NODE_CHILDREN_COUNT) {
                    e.emit(brick.clone())?;
                }
                Ok(())
            }),
            NodeContent::UniformLeaf(brick) => encoder.emit_list(|e| {
                e.emit_str("##u#")?;
                e.emit(brick.clone())
            }),
        }
    }
}

impl<T> FromBencode for NodeContent<T>
where
    T: FromBencode + Debug + Clone + PartialEq,
{
    fn decode_bencode_object(data: Object) -> Result<Self, bendy::decoding::Error> {
        match data {
            Object::List(mut list) => {
                let (is_leaf, is_uniform) = match list.next_object()?.unwrap() {
                    Object::Bytes(b) => {
                        match String::from_utf8(b.to_vec())
                            .unwrap_or("".to_string())
                            .as_str()
                        {
                            "##" => {
                                // The content is an internal Node
                                Ok((false, false))
                            }
                            "###" => {
                                // The content is a leaf
                                Ok((true, false))
                            }
                            "##u#" => {
                                // The content is a uniform leaf
                                Ok((true, true))
                            }
                            misc => Err(bendy::decoding::Error::unexpected_token(
                                "A NodeContent Identifier string, which is either # or ##",
                                "The string ".to_owned() + misc,
                            )),
                        }
                    }
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "A NodeContent Identifier, which is a string",
                        "Something else",
                    )),
                }?;

                debug_assert!(
                    is_leaf || is_uniform,
                    "Expected NodeContent list to be either leaf, uniform or both!"
                );

                if is_leaf && !is_uniform {
                    let leaf_data: [BrickData<T>; BOX_NODE_CHILDREN_COUNT] = (0
                        ..BOX_NODE_CHILDREN_COUNT)
                        .map(|_sectant| {
                            BrickData::decode_bencode_object(
                                list.next_object()
                                    .expect("Expected BrickData object:")
                                    .unwrap(),
                            )
                            .expect("Expected to decode BrickData:")
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();

                    return Ok(NodeContent::Leaf(leaf_data));
                }

                if is_leaf && is_uniform {
                    return Ok(NodeContent::UniformLeaf(BrickData::decode_bencode_object(
                        list.next_object()?.unwrap(),
                    )?));
                }
                panic!(
                    "The logical combination of !is_leaf and is_uniform should never be reached"
                );
            }
            Object::Bytes(b) => {
                // NodeContent is either Internal or Nothing
                match String::from_utf8(b.to_vec())
                    .unwrap_or("".to_string())
                    .as_str()
                {
                    "#" => Ok(NodeContent::Nothing),
                    "##" => Ok(NodeContent::Internal),
                    something_else => Err(bendy::decoding::Error::unexpected_token(
                        "Nodecontent node type to be Internal or Nothing,",
                        something_else,
                    )),
                }
            }
            _ => Err(bendy::decoding::Error::unexpected_token(
                "A NodeContent Object, either a List or a ByteString",
                "Something else",
            )),
        }
    }
}

//####################################################################################
//  ██████   █████    ███████    ██████████   ██████████
// ░░██████ ░░███   ███░░░░░███ ░░███░░░░███ ░░███░░░░░█
//  ░███░███ ░███  ███     ░░███ ░███   ░░███ ░███  █ ░
//  ░███░░███░███ ░███      ░███ ░███    ░███ ░██████
//  ░███ ░░██████ ░███      ░███ ░███    ░███ ░███░░█
//  ░███  ░░█████ ░░███     ███  ░███    ███  ░███ ░   █
//  █████  ░░█████ ░░░███████░   ██████████   ██████████
// ░░░░░    ░░░░░    ░░░░░░░    ░░░░░░░░░░   ░░░░░░░░░░
//    █████████  █████   █████ █████ █████       ██████████   ███████████   ██████████ ██████   █████
//   ███░░░░░███░░███   ░░███ ░░███ ░░███       ░░███░░░░███ ░░███░░░░░███ ░░███░░░░░█░░██████ ░░███
//  ███     ░░░  ░███    ░███  ░███  ░███        ░███   ░░███ ░███    ░███  ░███  █ ░  ░███░███ ░███
// ░███          ░███████████  ░███  ░███        ░███    ░███ ░██████████   ░██████    ░███░░███░███
// ░███          ░███░░░░░███  ░███  ░███        ░███    ░███ ░███░░░░░███  ░███░░█    ░███ ░░██████
// ░░███     ███ ░███    ░███  ░███  ░███      █ ░███    ███  ░███    ░███  ░███ ░   █ ░███  ░░█████
//  ░░█████████  █████   █████ █████ ███████████ ██████████   █████   █████ ██████████ █████  ░░█████
//   ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░ ░░░░░░░░░░░ ░░░░░░░░░░   ░░░░░   ░░░░░ ░░░░░░░░░░ ░░░░░    ░░░░░
//####################################################################################
// using generic arguments means the default key needs to be serialzied along with the data, which means a lot of wasted space..
// so serialization for the current ObjectPool key is adequate; The engineering hour cost of implementing new serialization logic
// every time the ObjectPool::Itemkey type changes is acepted.
impl ToBencode for NodeChildren<u32> {
    const MAX_DEPTH: usize = 2;
    fn encode(&self, encoder: SingleItemEncoder) -> Result<(), BencodeError> {
        match &self {
            NodeChildren::Children(c) => encoder.emit_list(|e| {
                e.emit_str("##c##")?;
                for child in c.iter().take(BOX_NODE_CHILDREN_COUNT) {
                    e.emit(child)?;
                }
                Ok(())
            }),
            NodeChildren::NoChildren => encoder.emit_str("##x##"),
        }
    }
}

impl FromBencode for NodeChildren<u32> {
    fn decode_bencode_object(data: Object) -> Result<Self, bendy::decoding::Error> {
        match data {
            Object::List(mut list) => {
                let marker = String::decode_bencode_object(list.next_object()?.unwrap())?;
                match marker.as_str() {
                    "##c##" => {
                        let mut c = Vec::new();
                        for _ in 0..BOX_NODE_CHILDREN_COUNT {
                            c.push(
                                u32::decode_bencode_object(list.next_object()?.unwrap())
                                    .ok()
                                    .unwrap(),
                            );
                        }
                        Ok(NodeChildren::Children(c.try_into().ok().unwrap()))
                    }
                    "##x##" => todo!(),
                    s => Err(bendy::decoding::Error::unexpected_token(
                        "A NodeChildren marker, either ##b## or ##c##",
                        s,
                    )),
                }
            }
            Object::Bytes(b) => {
                debug_assert_eq!(
                    String::from_utf8(b.to_vec())
                        .unwrap_or("".to_string())
                        .as_str(),
                    "##x##"
                );
                Ok(NodeChildren::default())
            }
            _ => Err(bendy::decoding::Error::unexpected_token(
                "A NodeChildren Object, Either a List or a ByteString",
                "Something else",
            )),
        }
    }
}

//####################################################################################
//  ██████   ██████ █████ ███████████
// ░░██████ ██████ ░░███ ░░███░░░░░███
//  ░███░█████░███  ░███  ░███    ░███
//  ░███░░███ ░███  ░███  ░██████████
//  ░███ ░░░  ░███  ░███  ░███░░░░░░
//  ░███      ░███  ░███  ░███
//  █████     █████ █████ █████
// ░░░░░     ░░░░░ ░░░░░ ░░░░░
//  ███████████ ██████████   █████████   ███████████ █████  █████ ███████████   ██████████  █████████
// ░░███░░░░░░█░░███░░░░░█  ███░░░░░███ ░█░░░███░░░█░░███  ░░███ ░░███░░░░░███ ░░███░░░░░█ ███░░░░░███
//  ░███   █ ░  ░███  █ ░  ░███    ░███ ░   ░███  ░  ░███   ░███  ░███    ░███  ░███  █ ░ ░███    ░░░
//  ░███████    ░██████    ░███████████     ░███     ░███   ░███  ░██████████   ░██████   ░░█████████
//  ░███░░░█    ░███░░█    ░███░░░░░███     ░███     ░███   ░███  ░███░░░░░███  ░███░░█    ░░░░░░░░███
//  ░███  ░     ░███ ░   █ ░███    ░███     ░███     ░███   ░███  ░███    ░███  ░███ ░   █ ███    ░███
//  █████       ██████████ █████   █████    █████    ░░████████   █████   █████ ██████████░░█████████
// ░░░░░       ░░░░░░░░░░ ░░░░░   ░░░░░    ░░░░░      ░░░░░░░░   ░░░░░   ░░░░░ ░░░░░░░░░░  ░░░░░░░░░
//####################################################################################
impl ToBencode for MIPMapStrategy {
    const MAX_DEPTH: usize = 3;
    fn encode(&self, encoder: SingleItemEncoder) -> Result<(), BencodeError> {
        encoder.emit_list(|e| {
            e.emit_int(self.enabled as u8)?;

            e.emit_int(self.resampling_methods.len())?;
            for entry in self.resampling_methods.iter() {
                e.emit(entry.0)?;
                e.emit(entry.1)?;
            }
            e.emit_int(self.resampling_color_matching_thresholds.len())?;
            for entry in self.resampling_color_matching_thresholds.iter() {
                e.emit(entry.0)?;
                e.emit_int((entry.1 * 1000.) as u32)?;
            }
            Ok(())
        })
    }
}

impl FromBencode for MIPMapStrategy {
    fn decode_bencode_object(data: Object) -> Result<Self, bendy::decoding::Error> {
        match data {
            Object::List(mut list) => {
                let enabled = match list.next_object()?.unwrap() {
                    Object::Integer("0") => Ok(false),
                    Object::Integer("1") => Ok(true),
                    Object::Integer(i) => Err(bendy::decoding::Error::unexpected_token(
                        "boolean field albedo_mip_maps",
                        format!("the number: {}", i),
                    )),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "boolean field albedo_mip_maps",
                        "Something else",
                    )),
                }?;

                let resampling_strategy_len = match list.next_object()?.unwrap() {
                    Object::Integer(i) => Ok(i.parse()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field MIP resampling strategy length",
                        "Something else",
                    )),
                }?;
                let mut resampling_methods = HashMap::new();
                for _ in 0..resampling_strategy_len {
                    let key = usize::decode_bencode_object(list.next_object()?.unwrap()).unwrap();
                    let value =
                        MIPResamplingMethods::decode_bencode_object(list.next_object()?.unwrap())
                            .unwrap();
                    resampling_methods.insert(key, value);
                }

                let resampling_strategy_len = match list.next_object()?.unwrap() {
                    Object::Integer(i) => Ok(i.parse::<usize>()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field MIP color matching strategy length",
                        "Something else",
                    )),
                }?;
                let mut resampling_color_matching_thresholds = HashMap::new();
                for _ in 0..resampling_strategy_len {
                    let key = usize::decode_bencode_object(list.next_object()?.unwrap()).unwrap();
                    let value = match list.next_object()?.unwrap() {
                        Object::Integer(i) => Ok(i.parse::<u32>()?),
                        _ => Err(bendy::decoding::Error::unexpected_token(
                            "int field MIP color matching strategy length",
                            "Something else",
                        )),
                    }?;
                    resampling_color_matching_thresholds.insert(key, value as f32 / 1000.);
                }

                Ok(Self {
                    enabled,
                    resampling_methods,
                    resampling_color_matching_thresholds,
                })
            }
            _ => Err(bendy::decoding::Error::unexpected_token("List", "not List")),
        }
    }
}

impl ToBencode for MIPResamplingMethods {
    const MAX_DEPTH: usize = 2;
    fn encode(&self, encoder: SingleItemEncoder) -> Result<(), BencodeError> {
        match self {
            MIPResamplingMethods::BoxFilter => encoder.emit_int(0),
            MIPResamplingMethods::PointFilter => encoder.emit_int(1),
            MIPResamplingMethods::PointFilterBD => encoder.emit_int(2),
            MIPResamplingMethods::Posterize(threshold) => {
                encoder.emit_int(3 + (threshold * 1000.) as u32)
            }
            MIPResamplingMethods::PosterizeBD(threshold) => {
                encoder.emit_int(1003 + (threshold * 1000.) as u32)
            }
        }
    }
}

impl FromBencode for MIPResamplingMethods {
    fn decode_bencode_object(data: Object) -> Result<Self, bendy::decoding::Error> {
        match data {
            Object::Integer("0") => Ok(MIPResamplingMethods::BoxFilter),
            Object::Integer("1") => Ok(MIPResamplingMethods::PointFilter),
            Object::Integer("2") => Ok(MIPResamplingMethods::PointFilterBD),
            Object::Integer(int) => match int
                .parse::<u32>()
                .unwrap_or_else(|_| panic!("Expected to be able to parse: {:?} as u32", int))
            {
                thr if (3..1002).contains(&thr) => {
                    Ok(MIPResamplingMethods::Posterize((thr as f32 - 3.) / 1000.))
                }

                thr if (1003..2001).contains(&thr) => Ok(MIPResamplingMethods::PosterizeBD(
                    (thr as f32 - 1003.) / 1000.,
                )),
                num => Err(bendy::decoding::Error::unexpected_token(
                    "Integer of Posterized Enum type ranges",
                    format!("the number: {num}").to_owned(),
                )),
            },
            _ => Err(bendy::decoding::Error::unexpected_token(
                "Integer of Enum type",
                "not that",
            )),
        }
    }
}

//####################################################################################
//     ███████      █████████  ███████████ ███████████   ██████████ ██████████
//   ███░░░░░███   ███░░░░░███░█░░░███░░░█░░███░░░░░███ ░░███░░░░░█░░███░░░░░█
//  ███     ░░███ ███     ░░░ ░   ░███  ░  ░███    ░███  ░███  █ ░  ░███  █ ░
// ░███      ░███░███             ░███     ░██████████   ░██████    ░██████
// ░███      ░███░███             ░███     ░███░░░░░███  ░███░░█    ░███░░█
// ░░███     ███ ░░███     ███    ░███     ░███    ░███  ░███ ░   █ ░███ ░   █
//  ░░░███████░   ░░█████████     █████    █████   █████ ██████████ ██████████
//    ░░░░░░░      ░░░░░░░░░     ░░░░░    ░░░░░   ░░░░░ ░░░░░░░░░░ ░░░░░░░░░░
//####################################################################################
const SERIALIZE_MAX_DEPTH: usize = 10;
impl<T> BoxTree<T>
where
    T: ToBencode + Default + Clone + Eq + Hash,
{
    /// The number of bytes to read from the bytes of an octree that makes sure
    /// that the version object is included in the included bytes
    pub(crate) fn bytes_until_version() -> usize {
        std::mem::size_of::<crate::Version>() * 2
    }

    pub(crate) fn parse_version(bytes: &[u8]) -> Result<crate::Version, Error> {
        match bendy::decoding::Decoder::new(bytes)
            .with_max_depth(SERIALIZE_MAX_DEPTH)
            .next_object()?
            .expect("Expected BoxTree object list")
        {
            Object::List(mut list) => crate::Version::decode_bencode_object(
                list.next_object()?.expect("Expected Version object bytes"),
            ),
            _ => Err(bendy::decoding::Error::unexpected_token("List", "not List")),
        }
    }
}

impl<T> ToBencode for BoxTree<T>
where
    T: ToBencode + Default + Clone + Eq + Hash,
{
    const MAX_DEPTH: usize = SERIALIZE_MAX_DEPTH;
    fn encode(&self, encoder: SingleItemEncoder) -> Result<(), BencodeError> {
        //TODO: encode/decode node data
        encoder.emit_list(|e| {
            e.emit(crate::version())?;
            e.emit_int(self.auto_simplify as u8)?;
            e.emit_int(self.boxtree_size)?;
            e.emit_int(self.brick_dim)?;
            e.emit(&self.nodes)?;
            e.emit(&self.voxel_color_palette)?;
            e.emit(&self.voxel_data_palette)?;
            e.emit(&self.mip_map_strategy)?;
            Ok(())
        })
    }
}

impl<T> FromBencode for BoxTree<T>
where
    T: FromBencode + Default + Clone + Eq + Hash,
{
    fn decode_bencode_object(data: Object) -> Result<Self, bendy::decoding::Error> {
        match data {
            Object::List(mut list) => {
                list.next_object()?.expect("Expected Version object string");
                let auto_simplify = match list.next_object()?.unwrap() {
                    Object::Integer("0") => Ok(false),
                    Object::Integer("1") => Ok(true),
                    Object::Integer(i) => Err(bendy::decoding::Error::unexpected_token(
                        "boolean field auto_simplify",
                        format!("the number: {}", i),
                    )),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "boolean field auto_simplify",
                        "Something else",
                    )),
                }?;

                let boxtree_size = match list.next_object()?.unwrap() {
                    Object::Integer(i) => Ok(i.parse()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field boxtree_size",
                        "Something else",
                    )),
                }?;

                let brick_dim = match list.next_object()?.unwrap() {
                    Object::Integer(i) => Ok(i.parse()?),
                    _ => Err(bendy::decoding::Error::unexpected_token(
                        "int field boxtree_size",
                        "Something else",
                    )),
                }?;

                let nodes = ObjectPool::decode_bencode_object(list.next_object()?.unwrap())?;

                let voxel_color_palette =
                    Vec::<Albedo>::decode_bencode_object(list.next_object()?.unwrap())?;
                let mut map_to_color_index_in_palette = HashMap::new();
                for (i, voxel_color) in voxel_color_palette.iter().enumerate() {
                    map_to_color_index_in_palette.insert(*voxel_color, i);
                }

                let voxel_data_palette =
                    Vec::<T>::decode_bencode_object(list.next_object()?.unwrap())?;
                let mut map_to_data_index_in_palette = HashMap::new();
                for (i, voxel_data) in voxel_data_palette.iter().enumerate() {
                    map_to_data_index_in_palette.insert(voxel_data.clone(), i);
                }

                let mip_map_strategy =
                    MIPMapStrategy::decode_bencode_object(list.next_object()?.unwrap())?;

                Ok(Self {
                    auto_simplify,
                    boxtree_size,
                    brick_dim,
                    nodes,
                    voxel_color_palette,
                    voxel_data_palette,
                    map_to_color_index_in_palette,
                    map_to_data_index_in_palette,
                    mip_map_strategy,
                    update_triggers: vec![], // Cannot serialize output triggers
                })
            }
            _ => Err(bendy::decoding::Error::unexpected_token("List", "not List")),
        }
    }
}
