"""Select original clip anchors directly from the organizer's raw archive."""
from __future__ import annotations
import hashlib
import re
import zipfile
from collections import defaultdict
from pathlib import PurePosixPath
import cv2
import numpy as np

SLOTS = (0, 8, 16, 24, 31)

def visual_members(archive: zipfile.ZipFile) -> dict:
    clips = defaultdict(lambda: defaultdict(list))
    for member in archive.infolist():
        if member.is_dir() or not member.filename.lower().endswith('.png'):
            continue
        parts = PurePosixPath(member.filename).parts
        for at, part in enumerate(parts[:-1]):
            if re.fullmatch(r'SM_test_\d{4}', part) and parts[at+1] in ('IR','Depth_Color'):
                clips[part][parts[at+1]].append(member)
                break
    for modalities in clips.values():
        for members in modalities.values():
            members.sort(key=lambda item: tuple(map(int,re.findall(r'\d+',PurePosixPath(item.filename).name))))
    return dict(clips)

def select_anchors(archive: zipfile.ZipFile, members: list, *, decode_images=False) -> dict:
    if not members:
        return dict(available_frames=0, usable=False, reason='missing_modality', anchors=[])
    picks = np.rint(np.linspace(0,len(members)-1,32)).astype(np.int64)
    decoded = {}
    anchors = []
    for slot in SLOTS:
        requested = int(picks[slot])
        for actual in sorted(range(len(members)),key=lambda i:(abs(i-requested),i)):
            if actual not in decoded:
                payload=archive.read(members[actual])
                frame=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_GRAYSCALE)
                decoded[actual]=(frame,hashlib.sha256(payload).hexdigest())
            frame,digest=decoded[actual]
            if frame is None:
                continue
            row=dict(slot=slot,source_index=requested,actual_source_index=actual,
                     member=members[actual].filename,payload_sha256=digest,
                     source_header_offset=members[requested].header_offset,
                     requested_bytes=members[requested].file_size)
            if decode_images: row['image']=frame
            anchors.append(row)
            break
        else:
            return dict(available_frames=0,usable=False,reason='unusable_modality',anchors=[])
    return dict(available_frames=len(members),usable=True,
                temporally_scannable=len({r['source_index'] for r in anchors})>=3,anchors=anchors)
