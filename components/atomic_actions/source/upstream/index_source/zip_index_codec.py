"""Original ZIP64 central-directory parser; no network or model imports."""
import struct

def parse_zip64_extra(extra, uncomp_size, comp_size, local_offset, disk_start):
    pos = 0
    while pos + 4 <= len(extra):
        field_id, field_len = struct.unpack_from("<HH", extra, pos)
        body = extra[pos + 4 : pos + 4 + field_len]
        pos += 4 + field_len
        if field_id != 0x0001:
            continue
        cursor = 0

        def take(fmt):
            nonlocal cursor
            size = struct.calcsize(fmt)
            if cursor + size > len(body):
                raise ValueError("truncated ZIP64 extra field")
            value = struct.unpack_from(fmt, body, cursor)[0]
            cursor += size
            return value

        if uncomp_size == 0xFFFFFFFF:
            uncomp_size = take("<Q")
        if comp_size == 0xFFFFFFFF:
            comp_size = take("<Q")
        if local_offset == 0xFFFFFFFF:
            local_offset = take("<Q")
        if disk_start == 0xFFFF:
            disk_start = take("<L")
        break
    return uncomp_size, comp_size, local_offset, disk_start

def parse_entries(data):
    pos = 0
    while pos + 46 <= len(data):
        if data[pos : pos + 4] != b"PK\x01\x02":
            raise ValueError(f"central directory signature missing at {pos}")
        values = struct.unpack_from("<4s6H3L5H2L", data, pos)
        flags = values[3]
        method = values[4]
        crc = values[7]
        comp_size = values[8]
        uncomp_size = values[9]
        name_len = values[10]
        extra_len = values[11]
        comment_len = values[12]
        disk_start = values[13]
        local_offset = values[16]
        name_start = pos + 46
        name_bytes = data[name_start : name_start + name_len]
        extra = data[
            name_start + name_len : name_start + name_len + extra_len
        ]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding, errors="replace")
        uncomp_size, comp_size, local_offset, disk_start = parse_zip64_extra(
            extra, uncomp_size, comp_size, local_offset, disk_start
        )
        yield {
            "name": name,
            "disk": int(disk_start),
            "offset": int(local_offset),
            "compressed": int(comp_size),
            "uncompressed": int(uncomp_size),
            "method": int(method),
            "flags": int(flags),
            "crc": int(crc),
        }
        pos += 46 + name_len + extra_len + comment_len
    if pos != len(data):
        raise ValueError(f"unparsed central-directory bytes: {len(data) - pos}")
