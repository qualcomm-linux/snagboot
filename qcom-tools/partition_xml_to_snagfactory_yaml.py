#!/usr/bin/env python3
# This file is part of Snagboot
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import argparse
import os
import re
import uuid
from typing import Any, Optional
from xml.etree import ElementTree as ET

import yaml

DEFAULT_SECTOR_SIZE = 512
DEFAULT_WRITE_PROTECT_BOUNDARY_IN_KB = 0
DEFAULT_WRITE_PROTECT_GPT_PARTITION_TABLE = False
DEFAULT_PERFORMANCE_BOUNDARY_IN_KB = 0
DEFAULT_ALIGN_PARTITIONS_TO_PERFORMANCE_BOUNDARY = False
DEFAULT_GROW_LAST_PARTITION = False

DEFAULT_ERASEBLK_SIZE = 0x40000

DEFAULT_QUPFW_PATH = "qupv3fw.elf"
DEFAULT_QUPFW_LUN = 4
DEFAULT_QUPFW_PART = "qupfw_a"

# ---------------------------------------------------------------------------
# Flash type detection (UFS vs SPI-NOR/MTD)
# ---------------------------------------------------------------------------


def detect_flash_type(xml_path: str) -> str:
    """Decides whether the input XML describes a UFS or an MTD/SPI-NOR
    partition layout, based on substrings in the XML's basename:
      - "ufs" in the name -> "ufs"  (scsi backend, "flash_image:<f>:<n>:<p>")
      - "nor" in the name -> "nor"  (mtd backend, "flash_image:<f>:mtd:<n>:<p>")
    where, f: file path
           n: device num
           p: partition name

    Raises a ValueError if neither substrings are found or both substrings
    are found
    """
    name = os.path.basename(xml_path).lower()
    has_ufs = "ufs" in name
    has_nor = "nor" in name

    if has_ufs and has_nor:
        raise ValueError(
            f"Cannot determine flash type from XML filename '{name}': "
            f"it matches both 'ufs' and 'nor' substrings, please rename "
            f"the input file so only one applies."
        )
    if has_ufs:
        return "ufs"
    if has_nor:
        return "nor"

    raise ValueError(
        f"Cannot determine flash type from XML filename '{name}': "
        f"expected an 'ufs' or 'nor' substring in the filename."
    )


# ---------------------------------------------------------------------------
# Firmware config (xbl / u-boot paths + image_ids) input YAML
# ---------------------------------------------------------------------------


def load_firmware_config(path: str) -> dict:
    """Loads the YAML file describing the xbl and u-boot firmware
    binaries to be used by snagrecover, e.g.:

        xbl:
          path: prog_snagboot_ddr.elf
          image_id: <img_id>
        u-boot:
          path: u-boot.mbn
          image_id: <img_id>
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Firmware config '{path}' must be a YAML mapping")

    for key in ("xbl", "u-boot"):
        if key not in data or not isinstance(data[key], dict):
            raise ValueError(
                f"Firmware config '{path}' is missing required '{key}' section"
            )
        section = data[key]
        if "path" not in section or "image_id" not in section:
            raise ValueError(
                f"Firmware config '{path}' section '{key}' must define "
                f"both 'path' and 'image_id'"
            )

    return data


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def parse_parser_instructions(root: ET.Element) -> dict:
    """Parses the <parser_instructions> block into a dict of
    raw string values.
    """
    instructions = {}
    node = root.find("parser_instructions")
    if node is None or node.text is None:
        return instructions

    for m in re.finditer(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s]+)", node.text
    ):
        instructions[m.group(1)] = m.group(2)

    return instructions


def to_bool(val: Optional[str], default: bool = False) -> bool:
    """Converts a raw XML attribute string to a bool, falling back to
    'default' if 'val' is None.
    """
    if val is None:
        return default
    return str(val).strip().lower() == "true"


def to_int(val: Optional[str], default: int = 0) -> int:
    """Converts a raw XML attribute string to an int, falling back to
    'default' if 'val' is None or empty.
    """
    if val is None or str(val).strip() == "":
        return default
    return int(val)


def parse_physical_partitions(
    root: ET.Element, default_performance_boundary_in_kb: int
) -> list:
    """Returns a list of LUNs; each LUN is a list of partition dicts, in
    XML order.
    """
    luns = []

    for phy_elem in root.findall("physical_partition"):
        partitions = []

        for part_elem in phy_elem.findall("partition"):
            partitions.append({
                "label": part_elem.get("label", ""),
                "type": part_elem.get("type", ""),
                "size_in_kb": to_int(part_elem.get("size_in_kb"), 0),
                "bootable": to_bool(part_elem.get("bootable")),
                "readonly": to_bool(part_elem.get("readonly")),
                "hidden": to_bool(part_elem.get("hidden")),
                "system": to_bool(part_elem.get("system")),
                "dontautomount": to_bool(part_elem.get("dontautomount")),
                "filename": part_elem.get("filename", ""),
                "tries_remaining": to_int(part_elem.get("triesremaining"), 0),
                "priority": to_int(part_elem.get("priority"), 0),
                "uguid": part_elem.get("uniqueguid"),
                # Per-partition override of the global
                "performance_boundary_in_kb": to_int(
                    part_elem.get("PERFORMANCE_BOUNDARY_IN_KB"),
                    default_performance_boundary_in_kb,
                ),
            })

        luns.append(partitions)

    return luns


def convert_kb_to_sectors(size_in_kb: int, sector_size: int) -> int:
    return int((size_in_kb * 1024) / sector_size)


def round_up_size_in_kb(size_in_kb: int, sector_size: int) -> int:
    """Rounds a partition's size (given in KB) up to a whole number of
    sectors. Returns the rounded size in bytes.
    """
    size_bytes = size_in_kb * 1024
    if size_bytes % sector_size > 0:
        sectors = (size_bytes // sector_size) + 1
        size_bytes = sectors * sector_size
    return size_bytes


def return_num_sectors_till_boundary(
    current_lba: int, boundary_in_kb: int, sector_size: int
) -> int:
    """Number of sectors that must be added to current_lba to land
    exactly on the next boundary_in_kb boundary (0 if already aligned
    or boundary disabled).
    """
    if boundary_in_kb <= 0:
        return 0

    boundary_sectors = convert_kb_to_sectors(boundary_in_kb, sector_size)
    remainder = current_lba % boundary_sectors
    if remainder > 0:
        return boundary_sectors - remainder
    return 0


def primary_gpt_reserved_sectors(sector_size: int) -> int:
    """Sectors reserved at the start of the disk for the protective
    MBR + primary GPT header + partition entry array:
        SECTOR_SIZE_IN_BYTES == 4096 -> 6 sectors   (1 + 1 + 4)
        otherwise (default 512)      -> 34 sectors  (1 + 1 + 32)
    """
    if sector_size == 4096:
        return 6
    return 34


# ---------------------------------------------------------------------------
# Write-protect boundary tracking
# ---------------------------------------------------------------------------


class WriteProtectTracker:
    """Write-protect "regions" are aligned chunks of
    WRITE_PROTECT_BOUNDARY_IN_KB sectors. Once a region is opened
    covering some sector range, it keeps growing (in units of the
    boundary) until it fully covers whatever readonly partition
    triggered/extended it. Partition placement then uses these regions
    to decide whether a partition needs to be moved:
      - a readonly partition may stay put if it's already covered by
        the current region (it will simply extend the region to fit)
      - a writable partition must be moved past the end of the current
        region if its start currently falls inside it (can't have
        writable data inside a locked region)
    """

    def __init__(
        self,
        write_protect_boundary_in_kb: int,
        sector_size: int,
        write_protect_gpt_partition_table: bool,
        first_lba: int,
    ) -> None:
        self.boundary_in_kb = write_protect_boundary_in_kb
        self.sector_size = sector_size
        self.boundary_sectors = (
            convert_kb_to_sectors(write_protect_boundary_in_kb, sector_size)
            if write_protect_boundary_in_kb > 0 else 0
        )

        self.regions = [{
            "start_sector": 0,
            "num_sectors": self.boundary_sectors,
            "end_sector": self.boundary_sectors - 1,
        }]

        if not write_protect_gpt_partition_table:
            self.regions[0]["num_sectors"] = 0
            self.regions[0]["end_sector"] = 0
        elif self.boundary_in_kb > 0:
            self.update(first_lba, 0)

    def end_sector(self) -> int:
        return self.regions[-1]["end_sector"]

    def sectors_till_boundary(self, current_lba: int) -> int:
        return return_num_sectors_till_boundary(
            current_lba, self.boundary_in_kb, self.sector_size
        )

    def update(self, start: int, size: int) -> None:
        """Grows the current region to cover [start, start+size-1], or opens
        a new region starting at 'start' if it isn't already covered.
        """
        if self.boundary_in_kb <= 0:
            return

        region = self.regions[-1]

        if start - 1 <= region["end_sector"]:
            while (start + size - 1) > region["end_sector"]:
                region["num_sectors"] += self.boundary_sectors
                region["end_sector"] += self.boundary_sectors
        else:
            new_region = {
                "start_sector": start,
                "num_sectors": self.boundary_sectors,
                "end_sector": start + self.boundary_sectors - 1,
            }
            self.regions.append(new_region)

            while (start + size - 1) > new_region["end_sector"]:
                new_region["num_sectors"] += self.boundary_sectors
                new_region["end_sector"] += self.boundary_sectors


def apply_write_protect_alignment(
    first_lba: int, part: dict, wp_tracker: WriteProtectTracker
) -> int:
    """readonly partitions may stay in place if already covered by
    the current WP region; writable partitions must be moved past the
    end of the current WP region if they currently fall inside it.
    """
    if wp_tracker.boundary_in_kb <= 0:
        return first_lba

    sectors_till_boundary = wp_tracker.sectors_till_boundary(first_lba)

    if part["readonly"]:
        if first_lba <= wp_tracker.end_sector():
            pass  # already covered by the current WP region
        else:
            first_lba += sectors_till_boundary
    else:
        if first_lba <= wp_tracker.end_sector():
            first_lba += sectors_till_boundary
        else:
            pass

    return first_lba


# ---------------------------------------------------------------------------
# Performance boundary alignment
# ---------------------------------------------------------------------------


def apply_performance_boundary_alignment(
    first_lba: int,
    part: dict,
    sector_size: int,
    global_performance_boundary_in_kb: int,
    align_to_performance_boundary: bool,
) -> int:
    partition_boundary_in_kb = part.get("performance_boundary_in_kb", 0)

    if global_performance_boundary_in_kb > 0:
        if not align_to_performance_boundary:
            print(
                f"WARNING: PERFORMANCE_BOUNDARY_IN_KB="
                f"{global_performance_boundary_in_kb} but "
                f"ALIGN_PARTITIONS_TO_PERFORMANCE_BOUNDARY is False; "
                f"partition '{part['label']}' will NOT be aligned to "
                f"this boundary."
            )
        else:
            sectors_till_boundary = return_num_sectors_till_boundary(
                first_lba, partition_boundary_in_kb, sector_size
            )
            if sectors_till_boundary > 0:
                first_lba += sectors_till_boundary
    else:
        if partition_boundary_in_kb > 0:
            print(
                f"partition '{part['label']}' is NOT aligned to a "
                f"performance boundary"
            )

    return first_lba


# ---------------------------------------------------------------------------
# GPT entry computation
# ---------------------------------------------------------------------------


def compute_attrs(part: dict) -> int:
    """
    bit 0  system
    bit 48 priority        (if > 0)
    bit 52 tries_remaining (if > 0)
    bit 60 readonly
    bit 62 hidden
    bit 63 dontautomount
    """
    attrs = 0

    if part["readonly"]:
        attrs |= 1 << 60
    if part["hidden"]:
        attrs |= 1 << 62
    if part["dontautomount"]:
        attrs |= 1 << 63
    if part["system"]:
        attrs |= 1
    if part["tries_remaining"] > 0:
        attrs |= part["tries_remaining"] << 52
    if part["priority"] > 0:
        attrs |= part["priority"] << 48

    return attrs


def get_partition_uuid(part: dict) -> str:
    """Reuses an explicit 'uniqueguid' XML attribute if present,
    otherwise falls back to a randomly generated UUID (uuid.uuid4())
    for UniquePartitionGUID.
    """
    if part.get("uguid"):
        return part["uguid"]
    return str(uuid.uuid4())


def get_disk_uuid(cli_uuid_disk: Optional[str] = None) -> str:
    """Returns the CLI-provided disk UUID if set, otherwise falls back
    to a randomly generated UUID (uuid.uuid4()).
    """
    if cli_uuid_disk:
        return cli_uuid_disk
    return str(uuid.uuid4())


def build_gpt_entries(
    partitions: list,
    sector_size: int,
    write_protect_boundary_in_kb: int,
    write_protect_gpt_partition_table: bool,
    performance_boundary_in_kb: int,
    align_to_performance_boundary: bool,
    grow_last_partition: bool,
    cli_uuid_disk: Optional[str] = None,
) -> tuple:
    """Computes start/size/uuid/type/attrs for every partition in a LUN."""
    entries = []
    first_lba = primary_gpt_reserved_sectors(sector_size)
    num_partitions = len(partitions)

    wp_tracker = WriteProtectTracker(
        write_protect_boundary_in_kb, sector_size,
        write_protect_gpt_partition_table, first_lba,
    )

    for idx, part in enumerate(partitions):
        is_last = (idx == num_partitions - 1)

        # Performance-boundary alignment
        first_lba = apply_performance_boundary_alignment(
            first_lba, part, sector_size,
            performance_boundary_in_kb, align_to_performance_boundary,
        )

        # Write-protect-boundary alignment
        first_lba = apply_write_protect_alignment(first_lba, part, wp_tracker)

        size_bytes = round_up_size_in_kb(part["size_in_kb"], sector_size)
        num_sectors = size_bytes // sector_size

        start_sector = first_lba
        start_bytes = start_sector * sector_size

        if is_last and grow_last_partition:
            size_field = "-"
        else:
            size_field = "0x%x" % size_bytes

        entries.append({
            "name": part["label"],
            "start": "0x%x" % start_bytes,
            "size": size_field,
            "attrs": "0x%x" % compute_attrs(part),
            "bootable": part["bootable"],
            "uuid": get_partition_uuid(part),
            "type": part["type"].lower(),
            "filename": part["filename"],
        })

        # If this partition is readonly, grow/open a WP region to
        # cover it
        if part["readonly"]:
            wp_tracker.update(start_sector, num_sectors)

        first_lba = start_sector + num_sectors

    disk_uuid = get_disk_uuid(cli_uuid_disk)
    return disk_uuid, entries


# ---------------------------------------------------------------------------
# Command / YAML generation
# ---------------------------------------------------------------------------


def format_gpt_partitions_string(disk_uuid: str, entries: list) -> str:
    """Builds the 'uuid_disk=...;name=...,start=...,...;...' string
    consumed by U-Boot's gpt write.
    """
    fields = ["uuid_disk=%s" % disk_uuid]

    for e in entries:
        parts = [
            "name=%s" % e["name"],
            "start=%s" % e["start"],
            "size=%s" % e["size"],
            "attrs=%s" % e["attrs"],
        ]
        if e["bootable"]:
            parts.append("bootable")
        parts.append("uuid=%s" % e["uuid"])
        parts.append("type=%s" % e["type"])
        fields.append(",".join(parts))

    return ";".join(fields) + ";"


def build_gpt_write_task_args(
    bus: str, dev: int, disk_uuid: str, entries: list
) -> str:
    """Returns the oem_run string for the 'run' task that writes
    the GPT table for this LUN/mtd device.
    """
    partitions_string = format_gpt_partitions_string(disk_uuid, entries)
    return "oem_run:gpt write %s %d '%s'" % (bus, dev, partitions_string)


def build_flash_image_args(bus: str, dev: int, entries: list) -> list:
    """Returns a list of 'flash_image:<file>:...' strings for every
    partition in this LUN/mtd device that has a filename set.

    For "scsi" (UFS) backends the syntax is:
        flash_image:<file>:<lun>:<part>
    For "mtd" (SPI-NOR) backends, an extra "mtd" field is inserted:
        flash_image:<file>:mtd:<dev>:<part>
    """
    args = []
    for e in entries:
        if not e["filename"]:
            continue
        if bus == "mtd":
            args.append(
                "flash_image:%s:mtd:%d:%s" % (e["filename"], dev, e["name"])
            )
        else:
            args.append(
                "flash_image:%s:%d:%s" % (e["filename"], dev, e["name"])
            )
    return args


def build_dev_tasks(
    bus: str, dev: int, disk_uuid: str, entries: list
) -> list:
    """Returns the list of snagfactory task dicts for one LUN (scsi) or
    mtd device (mtd): one 'run' task for the GPT write, one 'run' task
    for flashing images.
    """
    tasks = []

    tasks.append({
        "task": "run",
        "args": [build_gpt_write_task_args(bus, dev, disk_uuid, entries)],
    })

    flash_args = build_flash_image_args(bus, dev, entries)
    if flash_args:
        tasks.append({
            "task": "run",
            "args": flash_args,
        })

    return tasks


class HexInt(int):
    def __new__(cls, value):
        if isinstance(value, str):
            return super().__new__(cls, value, 0)
        return super().__new__(cls, value)


def build_snagfactory_config(
    flash_type: str,
    luns: list,
    sector_size: int,
    write_protect_boundary_in_kb: int,
    write_protect_gpt_partition_table: bool,
    performance_boundary_in_kb: int,
    align_to_performance_boundary: bool,
    grow_last_partition: bool,
    fw_config: dict,
    args: argparse.Namespace,
) -> dict:
    """Builds the full snagfactory YAML structure, for either a UFS
    ("ufs<n>" target-device) or SPI-NOR/MTD ("nor<n>" target-device).
    """
    bus = "scsi" if flash_type == "ufs" else "mtd"
    target_device_prefix = "ufs" if flash_type == "ufs" else "nor"

    globals_block = {
        "target-device": f"{target_device_prefix}0",
        "fb-buffer-addr": HexInt(args.fb_buffer_addr),
    }

    if flash_type == "nor":
        # Required by snagfactory for MTD target-device.
        globals_block["eraseblk-size"] = HexInt(args.eraseblk_size)

    if args.usb_wait_timeout is not None:
        globals_block["usb-wait-timeout"] = args.usb_wait_timeout

    tasks = [globals_block]

    for dev_index, partitions in enumerate(luns):
        if not partitions:
            continue

        disk_uuid, entries = build_gpt_entries(
            partitions, sector_size,
            write_protect_boundary_in_kb,
            write_protect_gpt_partition_table,
            performance_boundary_in_kb,
            align_to_performance_boundary,
            grow_last_partition, args.uuid_disk,
        )
        tasks.extend(build_dev_tasks(bus, dev_index, disk_uuid, entries))

    if flash_type == "ufs":
        # qupv3fw.elf must always be flashed to a scsi LUN before
        # any SPI-NOR flashing can proceed
        tasks.append({
            "task": "run",
            "args": [
                "flash_image:%s:%d:%s"
                % (args.qupfw_path, args.qupfw_lun, args.qupfw_part)
            ],
        })

    tasks.append({
        "task": "run",
        "args": ["reset"],
    })

    config = {
        "boards": {
            args.vidpid: args.soc_model,
        },
        "soc-models": {
            f"{args.soc_model}-firmware": {
                "xbl": {
                    "path": fw_config["xbl"]["path"],
                    "image_id": fw_config["xbl"]["image_id"],
                },
                "u-boot": {
                    "path": fw_config["u-boot"]["path"],
                    "image_id": fw_config["u-boot"]["image_id"],
                },
            },
            f"{args.soc_model}-tasks": tasks,
        },
    }
    return config


# ---------------------------------------------------------------------------
# YAML dumping
# ---------------------------------------------------------------------------


def dump_yaml(config: dict, out_path: str) -> None:
    class IndentedDumper(yaml.SafeDumper):
        def increase_indent(
            self,
            flow: bool = False,
            indentless: bool = False
        ):
            return super().increase_indent(flow, False)

    def hexint_presenter(dumper: yaml.SafeDumper, data: "HexInt"):
        return dumper.represent_scalar(
            "tag:yaml.org,2002:int", "0x%x" % int(data)
        )

    IndentedDumper.add_representer(HexInt, hexint_presenter)

    with open(out_path, "w") as f:
        yaml.dump(
            config, f, Dumper=IndentedDumper,
            sort_keys=False, default_flow_style=False,
        )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_selected_config(
    flash_type: str, args: argparse.Namespace, fw_config: dict, luns: list
) -> None:
    """Reports the configuration selected for flashing, so it's clear
    which target device / flash backend / firmware images will end up
    in the generated snagfactory YAML.
    """
    bus = "scsi" if flash_type == "ufs" else "mtd"
    target_device_prefix = "ufs" if flash_type == "ufs" else "nor"
    num_devices = sum(1 for partitions in luns if partitions)

    print("=" * 70)
    print("Snagfactory config generation summary")
    print("=" * 70)
    print(f"Input XML:            {args.xml_file}")
    print(
        f"Flash type selected:  "
        f"{'UFS' if flash_type == 'ufs' else 'SPI-NOR (mtd)'} "
        f"(detected from XML filename)"
    )
    print(f"U-Boot bus command:   {bus}")
    print(
        f"Target device(s):     "
        f"{', '.join(f'{target_device_prefix}{i}' for i in range(num_devices))}"
    )
    print(f"SoC model:            {args.soc_model}")
    print(f"USB vid:pid:          {args.vidpid}")
    print(f"fb-buffer-addr:       0x{HexInt(args.fb_buffer_addr):x}")
    if flash_type == "nor":
        print(f"eraseblk-size:        0x{HexInt(args.eraseblk_size):x}")

    if flash_type == "ufs":
        print(
            f"qupv3fw step:         flash_image:{args.qupfw_path}:"
            f"{args.qupfw_lun}:{args.qupfw_part}"
        )
    print(
        f"xbl firmware:         path={fw_config['xbl']['path']} "
        f"image_id={fw_config['xbl']['image_id']}"
    )
    print(
        f"u-boot firmware:      path={fw_config['u-boot']['path']} "
        f"image_id={fw_config['u-boot']['image_id']}"
    )
    print(f"Physical partitions:  {num_devices} device(s) parsed from XML")
    print(f"Output file:          {args.output}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def hex_or_dec_int(value: str) -> int:
    return int(value, 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a snagfactory YAML config from a "
                     "partition_ufs.xml or partition_nor.xml file"
    )
    parser.add_argument(
        "xml_file",
        help="Path to a partition XML file. The filename must contain "
             "'ufs' or 'nor' so the script can determine whether to "
             "generate a UFS (scsi) or SPI-NOR (mtd) snagfactory config.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output YAML path (default: "
             "'snagfactory-ufs.yaml' or 'snagfactory-nor.yaml' "
             "depending on the detected flash type)",
    )
    parser.add_argument("--vidpid", default="05c6:9008",
                        help="USB VID:PID")
    parser.add_argument(
        "--soc-model", required=True,
        help="snagrecover SoC model alias, e.g. iq9075",
    )
    parser.add_argument(
        "-f", "--firmware-config", required=True,
        help="Path to a YAML file describing the xbl and u-boot "
             "firmware images to recover with snagrecover",
    )
    parser.add_argument("--fb-buffer-addr", default="0xdb300000")
    parser.add_argument(
        "--usb-wait-timeout", type=int, default=30,
        help="Fastboot USB re-enumeration wait timeout in "
             "seconds, set higher than snagfactory's default "
             "(~9s) for SoCs whose U-Boot takes longer to boot "
             "and re-enumerate the Fastboot USB gadget "
             "(default: 30)",
    )
    parser.add_argument(
        "--uuid-disk", default=None,
        help="Override randomly generated disk UUID",
    )

    # SPI-NOR/MTD-specific options
    parser.add_argument(
        "--eraseblk-size", type=hex_or_dec_int, default=DEFAULT_ERASEBLK_SIZE,
        help="Erase block size in bytes for the MTD/SPI-NOR "
             f"'eraseblk-size' snagfactory global (default: "
             f"0x{DEFAULT_ERASEBLK_SIZE:x}). Only used when the flash "
             "type detected from the XML filename is SPI-NOR/MTD.",
    )
    parser.add_argument(
        "--qupfw-path", default=DEFAULT_QUPFW_PATH,
        help="Path to the qupv3fw.elf firmware image that must be "
             f"flashed before any SPI-NOR flashing (default: "
             f"'{DEFAULT_QUPFW_PATH}'). Only used for SPI-NOR/MTD output.",
    )
    parser.add_argument(
        "--qupfw-lun", type=int, default=DEFAULT_QUPFW_LUN,
        help="scsi LUN to which qupv3fw.elf must be flashed (default: "
             f"{DEFAULT_QUPFW_LUN}). Only used for SPI-NOR/MTD output.",
    )
    parser.add_argument(
        "--qupfw-part", default=DEFAULT_QUPFW_PART,
        help="Partition name to which qupv3fw.elf must be flashed "
             f"(default: '{DEFAULT_QUPFW_PART}'). Only used for "
             "SPI-NOR/MTD output.",
    )

    args = parser.parse_args()

    flash_type = detect_flash_type(args.xml_file)

    if args.output is None:
        args.output = f"snagfactory-{flash_type}.yaml"

    fw_config = load_firmware_config(args.firmware_config)

    tree = ET.parse(args.xml_file)
    root = tree.getroot()

    instructions = parse_parser_instructions(root)
    sector_size = to_int(
        instructions.get("SECTOR_SIZE_IN_BYTES"),
        DEFAULT_SECTOR_SIZE
    )
    write_protect_boundary_in_kb = to_int(
        instructions.get("WRITE_PROTECT_BOUNDARY_IN_KB"),
        DEFAULT_WRITE_PROTECT_BOUNDARY_IN_KB,
    )
    write_protect_gpt_partition_table = to_bool(
        instructions.get("WRITE_PROTECT_GPT_PARTITION_TABLE"),
        DEFAULT_WRITE_PROTECT_GPT_PARTITION_TABLE,
    )
    performance_boundary_in_kb = to_int(
        instructions.get("PERFORMANCE_BOUNDARY_IN_KB"),
        DEFAULT_PERFORMANCE_BOUNDARY_IN_KB,
    )
    align_to_performance_boundary = to_bool(
        instructions.get("ALIGN_PARTITIONS_TO_PERFORMANCE_BOUNDARY"),
        DEFAULT_ALIGN_PARTITIONS_TO_PERFORMANCE_BOUNDARY,
    )
    grow_last_partition = to_bool(
        instructions.get("GROW_LAST_PARTITION_TO_FILL_DISK"),
        DEFAULT_GROW_LAST_PARTITION,
    )

    luns = parse_physical_partitions(root, performance_boundary_in_kb)

    log_selected_config(flash_type, args, fw_config, luns)

    config = build_snagfactory_config(
        flash_type, luns, sector_size,
        write_protect_boundary_in_kb,
        write_protect_gpt_partition_table,
        performance_boundary_in_kb,
        align_to_performance_boundary,
        grow_last_partition, fw_config, args,
    )

    dump_yaml(config, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()