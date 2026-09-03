value = open("/workspace/.env", "rb").read()
token = value.split(b"=", 1)[1].rstrip(b"\n")
print({
    "file_bytes": len(value),
    "token_length": len(token),
    "prefix_ok": token.startswith(b"hf_"),
    "last_byte": token[-1],
    "contains_cr": b"\r" in token,
})
