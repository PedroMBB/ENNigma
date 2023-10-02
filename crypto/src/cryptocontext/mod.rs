use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use spin::Once;
use std::fmt::Debug;
use tfhe::boolean::prelude::*;

pub struct EncryptedContext {
    pub(crate) server_key_str: String,
    pub(crate) client_key: Option<ClientKey>,
    pub(crate) server_key: Once<ServerKey>,
}
impl Clone for EncryptedContext {
    fn clone(&self) -> Self {
        Self {
            server_key_str: self.server_key_str.to_string(),
            client_key: self.client_key.clone(),
            server_key: match self.server_key.get() {
                None => Once::new(),
                Some(b) => Once::initialized(b.clone()),
            },
        }
    }
}
impl Debug for EncryptedContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "EncryptedContext {{ server_key: true, client_key: {} }}",
            self.client_key.is_some()
        ))
    }
}
impl EncryptedContext {
    pub fn get_key(&self) -> &ServerKey {
        &self.server_key.call_once(|| {
            println!("Decoding server key");
            let server_key: Vec<u8> = general_purpose::STANDARD_NO_PAD
                .decode(&self.server_key_str)
                .expect("Should be able to decode the server_key");

            bincode::deserialize(server_key.as_slice()).expect("Should deserialize the server_key")
        })
    }
    pub fn remove_server_key(&self) -> Self {
        Self {
            server_key_str: self.server_key_str.clone(),
            server_key: Once::new(),
            client_key: self.client_key.clone(),
        }
    }
    // pub fn get_public_key(&self) -> &PublicKey {
    //     &self.public_key.call_once(|| {
    //         println!("Decoding public key");
    //         let public_key: Vec<u8> = general_purpose::STANDARD_NO_PAD
    //             .decode(&self.public_key_str)
    //             .expect("Should be able to decode the public_key");

    //         bincode::deserialize(&public_key).expect("Should deserialize the public_key")
    //     })
    // }
}

impl Serialize for EncryptedContext {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        pub struct SerObj<'a> {
            server_key: &'a str,
            client_key: Option<&'a ClientKey>,
        }

        (SerObj {
            client_key: self.client_key.as_ref(),
            server_key: &self.server_key_str,
        })
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for EncryptedContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        pub struct SerObj {
            server_key: String,
            client_key: Option<ClientKey>,
        }
        let v = SerObj::deserialize(deserializer)?;

        Ok(Self {
            server_key: Once::new(),
            server_key_str: v.server_key,
            client_key: v.client_key,
        })
    }
}

#[cfg(feature = "x86_64")]
impl Default for EncryptedContext {
    fn default() -> Self {
        Self::new()
    }
}

impl EncryptedContext {
    // #[cfg(feature = "wasm")]
    // pub fn new() -> Self {
    //     let ck = ClientKey::new(&DEFAULT_PARAMETERS);
    //     let sk = Ser::new(&ck);

    //     Self {
    //         server_key: None,
    //         public_key: Some(ck),
    //         client_key: Some(sk),
    //     }
    // }
    #[cfg(feature = "x86_64")]
    pub fn new() -> Self {
        let (client_key, server_key) = gen_keys();

        let server_key_str: Vec<u8> =
            bincode::serialize(&server_key).expect("Shoud be able to serialize server key");

        Self {
            server_key_str: general_purpose::STANDARD_NO_PAD.encode(server_key_str),
            server_key: Once::initialized(server_key),
            client_key: Some(client_key),
        }
    }

    pub fn get_server_context(&self) -> Self {
        Self {
            client_key: None,
            server_key_str: self.server_key_str.to_string(),
            server_key: Once::new(),
        }
    }
}
