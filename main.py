import collections
import struct
import sys
import math
import random

DEBUG_PRINT = False

NOP_INSTRUCTION = {'op': 'NOP', 'rd': 0, 'rs1': 0, 'rs2': 0, 'rs3': 0, 'imm': 0, 'funct3': 0, 'funct7': 0, 'is_fp': False, 'is_fp_dest': False, 'raw_hex': 0x00000013, 'raw_str': 'NOP', 'pc': -1}
MEM_SIZE = 1024 * 10
FLOAT_ZERO = 0.0
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
UINT64_MAX = 2**64 - 1
UINT32_MAX = 2**32 - 1

def double_to_int64(f):
    try:
        if math.isnan(f):
            return 0x7ff8000000000000
        if math.isinf(f):
            return 0x7ff0000000000000 if f > 0 else 0xfff0000000000000
        return struct.unpack('<q', struct.pack('<d', f))[0]
    except (struct.error, TypeError, OverflowError):
        return 0x7ff8000000000000

def int64_to_double(i):
    try:
        return struct.unpack('<d', struct.pack('<q', i))[0]
    except (struct.error, OverflowError, TypeError):
        return FLOAT_ZERO

FCLASS_NEG_INF   = 1 << 0
FCLASS_NEG_NORM  = 1 << 1
FCLASS_NEG_SUBN  = 1 << 2
FCLASS_NEG_ZERO  = 1 << 3
FCLASS_POS_ZERO  = 1 << 4
FCLASS_POS_SUBN  = 1 << 5
FCLASS_POS_NORM  = 1 << 6
FCLASS_POS_INF   = 1 << 7
FCLASS_SNAN      = 1 << 8
FCLASS_QNAN      = 1 << 9

def classify_double(d):
    if math.isnan(d):
        return FCLASS_QNAN
    elif math.isinf(d):
        return FCLASS_POS_INF if d > 0 else FCLASS_NEG_INF
    elif d == 0.0:
        return FCLASS_NEG_ZERO if math.copysign(1.0, d) < 0 else FCLASS_POS_ZERO
    else:
        SMALLEST_NORMAL = 2**-1022
        if 0 < abs(d) < SMALLEST_NORMAL:
             return FCLASS_POS_SUBN if d > 0 else FCLASS_NEG_SUBN
        else:
             return FCLASS_POS_NORM if d > 0 else FCLASS_NEG_NORM

def round_to_integer(f):
    if math.isnan(f) or math.isinf(f):
        raise ValueError("Cannot convert NaN or Inf to integer")
    return int(round(f))

def truncate_to_integer(f):
    if math.isnan(f) or math.isinf(f):
         raise ValueError("Cannot convert NaN or Inf to integer")
    return int(f)

class Processor:
    def __init__(self, instructions_in):
        self.pc = 0
        self.clock = 0
        self.regs = collections.defaultdict(int)
        self.fpregs = collections.defaultdict(float)
        self.data_mem = collections.defaultdict(int)
        self.instr_mem = {}
        self.labels = {}
        current_addr = 0
        line_num = 0
        self.raw_instructions = instructions_in

        for instr_str in instructions_in:
            line_num += 1
            instr_str_clean = instr_str.strip()
            if not instr_str_clean or instr_str_clean.startswith('#'):
                continue
            if ':' in instr_str_clean:
                label_part, _, instr_part = instr_str_clean.partition(':')
                label = label_part.strip()
                if label:
                    self.labels[label] = current_addr
                instr_str_clean = instr_part.strip()
                if not instr_str_clean: continue

            parsed = self.parse_instruction(instr_str_clean, current_addr, line_num)
            if parsed and parsed.get('op') != 'Unsupported':
                self.instr_mem[current_addr] = parsed
                current_addr += 4
            elif parsed and parsed.get('op') == 'Unsupported':
                 print(f"Warning: Unsupported or invalid instruction at line {line_num}: {instr_str_clean}")

        self.IF_ID = NOP_INSTRUCTION.copy()
        self.ID_EX = NOP_INSTRUCTION.copy()
        self.EX_MEM = NOP_INSTRUCTION.copy()
        self.MEM_WB = NOP_INSTRUCTION.copy()

        self.ID_EX.update({'rs1_val': 0, 'rs2_val': 0, 'rs3_val': 0, 'fs1_val': FLOAT_ZERO, 'fs2_val': FLOAT_ZERO, 'fs3_val': FLOAT_ZERO})
        self.EX_MEM.update({'alu_result': 0, 'fpu_result': FLOAT_ZERO, 'rs2_val_mem': 0, 'fs2_val_mem': FLOAT_ZERO, 'write_reg': -1, 'write_fp_reg': -1, 'mem_write': False, 'fp_mem_write': False, 'mem_read': False, 'fp_mem_read': False, 'reg_write': False, 'fp_reg_write': False, 'branch_taken': False, 'jump_target': -1})
        self.MEM_WB.update({'mem_data': 0, 'fp_mem_data': FLOAT_ZERO, 'alu_result_wb': 0, 'fpu_result_wb': FLOAT_ZERO, 'write_reg': -1, 'write_fp_reg': -1, 'reg_write': False, 'fp_reg_write': False, 'mem_read': False, 'fp_mem_read': False})

        self.stall = False
        self.flush = False
        self.branch_target_pc = -1
        self.instructions_executed = 0
        self.cycles = 0

    def parse_instruction(self, instr_str_raw, addr, line_num):
        instr_str_clean = instr_str_raw.strip().lower()
        if not instr_str_clean: return None
        parts_raw = instr_str_clean.replace(",", " ").replace("(", " ").replace(")", "").split()
        parts = [p for p in parts_raw if p]
        if not parts: return None

        op = parts[0].upper()
        instr = NOP_INSTRUCTION.copy()
        instr['op'] = op
        instr['raw_str'] = instr_str_raw
        instr['pc'] = addr
        instr['is_fp'] = False
        instr['is_fp_dest'] = False

        def parse_reg(reg_str, reg_type='any'):
            reg_str = reg_str.lower()
            if reg_str.startswith('x') and reg_type in ['any', 'int']:
                try: return int(reg_str[1:])
                except ValueError: return -1
            elif reg_str.startswith('f') and reg_type in ['any', 'fp']:
                try: return int(reg_str[1:])
                except ValueError: return -1
            else: return -1

        def parse_imm(imm_str):
            try: return int(imm_str, 0)
            except ValueError: return None

        try:
            if op in ['ADD', 'SUB', 'SLL', 'SLT', 'SLTU', 'XOR', 'SRL', 'SRA', 'OR', 'AND']:
                 instr['rd'] = parse_reg(parts[1], 'int'); instr['rs1'] = parse_reg(parts[2], 'int'); instr['rs2'] = parse_reg(parts[3], 'int')
            elif op in ['ADDI', 'SLTI', 'SLTIU', 'XORI', 'ORI', 'ANDI']:
                 instr['rd'] = parse_reg(parts[1], 'int'); instr['rs1'] = parse_reg(parts[2], 'int'); imm_val = parse_imm(parts[3]); instr['imm'] = imm_val if imm_val is not None else None
            elif op in ['SLLI', 'SRLI', 'SRAI']:
                 instr['rd'] = parse_reg(parts[1], 'int'); instr['rs1'] = parse_reg(parts[2], 'int'); imm_val = parse_imm(parts[3]); instr['imm'] = imm_val if imm_val is not None else None
            elif op == 'LW':
                 instr['rd'] = parse_reg(parts[1], 'int'); imm_val = parse_imm(parts[2]); instr['imm'] = imm_val if imm_val is not None else None; instr['rs1'] = parse_reg(parts[3], 'int')
            elif op == 'SW':
                 instr['rs2'] = parse_reg(parts[1], 'int'); imm_val = parse_imm(parts[2]); instr['imm'] = imm_val if imm_val is not None else None; instr['rs1'] = parse_reg(parts[3], 'int')
            elif op in ['BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU']:
                 instr['rs1'] = parse_reg(parts[1], 'int'); instr['rs2'] = parse_reg(parts[2], 'int'); instr['imm'] = parts[3]
            elif op == 'JAL':
                 instr['rd'] = parse_reg(parts[1], 'int'); instr['imm'] = parts[2]
            elif op == 'JALR':
                 instr['rd'] = parse_reg(parts[1], 'int'); imm_val = parse_imm(parts[2]); instr['imm'] = imm_val if imm_val is not None else None; instr['rs1'] = parse_reg(parts[3], 'int')
            elif op == 'FLD':
                 instr['rd'] = parse_reg(parts[1], 'fp'); imm_val = parse_imm(parts[2]); instr['imm'] = imm_val if imm_val is not None else None; instr['rs1'] = parse_reg(parts[3], 'int'); instr['is_fp'] = True; instr['is_fp_dest'] = True
            elif op == 'FSD':
                 instr['rs2'] = parse_reg(parts[1], 'fp'); imm_val = parse_imm(parts[2]); instr['imm'] = imm_val if imm_val is not None else None; instr['rs1'] = parse_reg(parts[3], 'int'); instr['is_fp'] = True; instr['is_fp_dest'] = False
            elif op in ['FADD.D', 'FSUB.D', 'FMUL.D', 'FDIV.D', 'FMIN.D', 'FMAX.D']:
                 instr['rd'] = parse_reg(parts[1], 'fp'); instr['rs1'] = parse_reg(parts[2], 'fp'); instr['rs2'] = parse_reg(parts[3], 'fp'); instr['is_fp'] = True; instr['is_fp_dest'] = True
            elif op == 'FSQRT.D':
                 instr['rd'] = parse_reg(parts[1], 'fp'); instr['rs1'] = parse_reg(parts[2], 'fp'); instr['is_fp'] = True; instr['is_fp_dest'] = True
            elif op in ['FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D']:
                 instr['rd'] = parse_reg(parts[1], 'fp'); instr['rs1'] = parse_reg(parts[2], 'fp'); instr['rs2'] = parse_reg(parts[3], 'fp'); instr['rs3'] = parse_reg(parts[4], 'fp'); instr['is_fp'] = True; instr['is_fp_dest'] = True
            elif op in ['FCVT.W.D', 'FCVT.WU.D', 'FCVT.L.D', 'FCVT.LU.D']:
                 instr['rd'] = parse_reg(parts[1], 'int'); instr['rs1'] = parse_reg(parts[2], 'fp'); instr['is_fp'] = True; instr['is_fp_dest'] = False
            elif op in ['FCVT.D.W', 'FCVT.D.WU', 'FCVT.D.L', 'FCVT.D.LU']:
                 instr['rd'] = parse_reg(parts[1], 'fp'); instr['rs1'] = parse_reg(parts[2], 'int'); instr['is_fp'] = False; instr['is_fp_dest'] = True
            elif op == 'FCVT.S.D':
                 instr['rd'] = parse_reg(parts[1], 'fp'); instr['rs1'] = parse_reg(parts[2], 'fp'); instr['is_fp'] = True; instr['is_fp_dest'] = True; instr['op'] = 'Unsupported';
            elif op == 'FCVT.D.S':
                 instr['rd'] = parse_reg(parts[1], 'fp'); instr['rs1'] = parse_reg(parts[2], 'fp'); instr['is_fp'] = True; instr['is_fp_dest'] = True; instr['op'] = 'Unsupported';
            elif op == 'FMV.X.D':
                 instr['rd'] = parse_reg(parts[1], 'int'); instr['rs1'] = parse_reg(parts[2], 'fp'); instr['is_fp'] = True; instr['is_fp_dest'] = False
            elif op == 'FMV.D.X':
                 instr['rd'] = parse_reg(parts[1], 'fp'); instr['rs1'] = parse_reg(parts[2], 'int'); instr['is_fp'] = False; instr['is_fp_dest'] = True
            elif op in ['FEQ.D', 'FLT.D', 'FLE.D']:
                 instr['rd'] = parse_reg(parts[1], 'int'); instr['rs1'] = parse_reg(parts[2], 'fp'); instr['rs2'] = parse_reg(parts[3], 'fp'); instr['is_fp'] = True; instr['is_fp_dest'] = False
            elif op == 'FCLASS.D':
                 instr['rd'] = parse_reg(parts[1], 'int'); instr['rs1'] = parse_reg(parts[2], 'fp'); instr['is_fp'] = True; instr['is_fp_dest'] = False
            elif op in ['FSGNJ.D', 'FSGNJN.D', 'FSGNJX.D']:
                 instr['rd'] = parse_reg(parts[1], 'fp'); instr['rs1'] = parse_reg(parts[2], 'fp'); instr['rs2'] = parse_reg(parts[3], 'fp'); instr['is_fp'] = True; instr['is_fp_dest'] = True
            elif op == 'NOP': pass
            else:
                instr['op'] = 'Unsupported'

            required_regs = []
            dest_reg = instr.get('rd', None)
            src1_reg = instr.get('rs1', None)
            src2_reg = instr.get('rs2', None)
            src3_reg = instr.get('rs3', None)

            if dest_reg is not None: required_regs.append(dest_reg)
            if src1_reg is not None: required_regs.append(src1_reg)
            if src2_reg is not None: required_regs.append(src2_reg)
            if src3_reg is not None: required_regs.append(src3_reg)

            if any(r == -1 for r in required_regs):
                instr['op'] = 'Unsupported'

            numeric_imm_ops = ['ADDI', 'SLTI', 'SLTIU', 'XORI', 'ORI', 'ANDI', 'SLLI', 'SRLI', 'SRAI', 'LW', 'SW', 'JALR', 'FLD', 'FSD']
            if op in numeric_imm_ops and instr.get('imm') is None:
                 instr['op'] = 'Unsupported'

            label_ops = ['BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU', 'JAL']
            if op in label_ops and not isinstance(instr.get('imm'), str):
                instr['op'] = 'Unsupported'

        except (IndexError, ValueError, TypeError) as e:
             print(f"Parser Exception: Error parsing '{instr_str_raw}' at line {line_num}: {e}")
             instr = NOP_INSTRUCTION.copy()
             instr['op'] = 'Unsupported'
             instr['raw_str'] = f"Parse Error: {instr_str_raw}"

        instr['funct3'] = instr.get('funct3', 0)
        instr['funct7'] = instr.get('funct7', 0)
        return instr

    def print_pipeline_state(self):
        print(f"Cycle {self.clock:<3} PC: {self.pc:<3} {'(Stalled)' if self.stall else ''}{'(Flushing)' if self.flush else ''}")

        if_id_pc = self.IF_ID.get('pc', -1)
        if if_id_pc == -1: print("  IF/ID : NOP")
        else: print(f"  IF/ID : PC={if_id_pc:<3} | {self.IF_ID.get('raw_str', 'Invalid').strip()}")

        id_ex_pc = self.ID_EX.get('pc', -1)
        if id_ex_pc == -1: print("  ID/EX : NOP")
        else:
            id_ex_instr = self.ID_EX.get('raw_str', 'Inv').strip(); id_ex_op = self.ID_EX.get('op', '?')
            rs1_v = self.ID_EX.get('rs1_val', 0); rs2_v = self.ID_EX.get('rs2_val', 0); rs3_v = self.ID_EX.get('rs3_val', 0)
            fs1_v = self.ID_EX.get('fs1_val', 0.0); fs2_v = self.ID_EX.get('fs2_val', 0.0); fs3_v = self.ID_EX.get('fs3_val', 0.0)
            imm = self.ID_EX.get('imm', 0); is_fp_dest = self.ID_EX.get('is_fp_dest', False)

            details = f"PC={id_ex_pc:<3} | {id_ex_instr:<25}"
            op1_str, op2_str, op3_str = "", "", ""

            op_needs_rs1_val = id_ex_op not in ['JAL', 'NOP', 'Unsupported']
            op_needs_rs2_val = id_ex_op in ['ADD','SUB','SLL','SLT','SLTU','XOR','SRL','SRA','OR','AND','SW','BEQ','BNE','BLT','BGE','BLTU','BGEU']
            op_needs_rs3_val = False
            op_needs_fs1_val = id_ex_op in ['FADD.D','FSUB.D','FMUL.D','FDIV.D', 'FMIN.D', 'FMAX.D', 'FSQRT.D', 'FCVT.W.D', 'FCVT.WU.D', 'FCVT.L.D', 'FCVT.LU.D', 'FEQ.D', 'FLT.D', 'FLE.D', 'FCLASS.D', 'FSGNJ.D', 'FSGNJN.D', 'FSGNJX.D', 'FMV.X.D', 'FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D']
            op_needs_fs2_val = id_ex_op in ['FADD.D','FSUB.D','FMUL.D','FDIV.D', 'FMIN.D', 'FMAX.D', 'FSD', 'FEQ.D', 'FLT.D', 'FLE.D', 'FSGNJ.D', 'FSGNJN.D', 'FSGNJX.D', 'FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D']
            op_needs_fs3_val = id_ex_op in ['FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D']
            op_needs_int_rs1_val = id_ex_op in ['LW', 'SW', 'JALR', 'FLD', 'FSD', 'FCVT.D.W', 'FCVT.D.WU', 'FCVT.D.L', 'FCVT.D.LU', 'FMV.D.X']

            if op_needs_fs1_val: op1_str = f"FS1_Val={fs1_v:<8.3f}"
            elif op_needs_int_rs1_val or op_needs_rs1_val: op1_str = f"RS1_Val=0x{rs1_v:<8x}"

            if op_needs_fs2_val: op2_str = f"FS2_Val={fs2_v:<8.3f}"
            elif op_needs_rs2_val: op2_str = f"RS2_Val=0x{rs2_v:<8x}"

            if op_needs_fs3_val: op3_str = f"FS3_Val={fs3_v:<8.3f}"
            elif op_needs_rs3_val: op3_str = f"RS3_Val=0x{rs3_v:<8x}"

            details += f" | {op1_str} {op2_str} {op3_str}".strip()

            numeric_imm_ops = ['ADDI','SLTI','SLTIU','XORI','ORI','ANDI','LW','SW','JALR','FLD','FSD', 'SLLI','SRLI','SRAI']
            label_imm_ops = ['BEQ','BNE','BLT','BGE','BLTU','BGEU', 'JAL']
            if id_ex_op in numeric_imm_ops and isinstance(imm, int): details += f" Imm=0x{imm:<4x}"
            elif id_ex_op in label_imm_ops and isinstance(imm, str): details += f" Imm='{str(imm)}'"

            print(f"  ID/EX : {details}")

        ex_mem_pc = self.EX_MEM.get('pc', -1)
        if ex_mem_pc == -1: print("  EX/MEM: NOP")
        else:
            ex_mem_instr = self.EX_MEM.get('raw_str', 'Inv').strip(); ex_mem_op = self.EX_MEM.get('op', '?')
            alu_r = self.EX_MEM.get('alu_result', 0); fpu_r = self.EX_MEM.get('fpu_result', 0.0)
            is_fp_dest = self.EX_MEM.get('is_fp_dest', False)
            wr_r = self.EX_MEM.get('write_reg', -1); wr_f = self.EX_MEM.get('write_fp_reg', -1)
            mr = self.EX_MEM.get('mem_read', False); mw = self.EX_MEM.get('mem_write', False)
            fmr = self.EX_MEM.get('fp_mem_read', False); fmw = self.EX_MEM.get('fp_mem_write', False)
            reg_w = self.EX_MEM.get('reg_write', False); fp_reg_w = self.EX_MEM.get('fp_reg_write', False)
            store_val_int = self.EX_MEM.get('rs2_val_mem', 0); store_val_fp = self.EX_MEM.get('fs2_val_mem', 0.0)

            details = f"PC={ex_mem_pc:<3} | {ex_mem_instr:<25}"
            dest_str = "-"
            result_str = ""

            if ex_mem_op in ['FADD.D', 'FSUB.D', 'FMUL.D', 'FDIV.D', 'FSQRT.D', 'FMIN.D', 'FMAX.D',
                             'FCVT.D.W', 'FCVT.D.WU', 'FCVT.D.L', 'FCVT.D.LU', 'FMV.D.X',
                             'FSGNJ.D', 'FSGNJN.D', 'FSGNJX.D', 'FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D', 'FLD']:
                result_str = f"FPU_Res={fpu_r:<8.3f}"
            else:
                result_str = f"ALU_Res=0x{alu_r:<8x}"

            details += f" | {result_str}"

            if reg_w and wr_r != -1: dest_str = f"x{wr_r:<2}"
            elif fp_reg_w and wr_f != -1: dest_str = f"f{wr_f:<2}"

            details += f" Dest={dest_str}"

            mem_op_str = "-"
            if mr: mem_op_str = f"MemRead (A=0x{alu_r:x})"
            elif mw: mem_op_str = f"MemWrite(A=0x{alu_r:x} <= 0x{store_val_int:x})"
            elif fmr: mem_op_str = f"FMemRead(A=0x{alu_r:x})"
            elif fmw: mem_op_str = f"FMemWrite(A=0x{alu_r:x} <= {store_val_fp:.3f})"
            details += f" | MemOp={mem_op_str:<30}"

            ctrls = f"{( 'RegW ' if reg_w else '')}{( 'FRegW ' if fp_reg_w else '')}{( 'MemR ' if mr else '')}{( 'FMemR ' if fmr else '')}{( 'MemW ' if mw else '')}{( 'FMemW ' if fmw else '')}".strip()
            details += f" | Ctrls=[{ctrls}]"
            print(f"  EX/MEM: {details}")

        mem_wb_pc = self.MEM_WB.get('pc', -1)
        if mem_wb_pc == -1: print("  MEM/WB: NOP")
        else:
            mem_wb_instr = self.MEM_WB.get('raw_str', 'Inv').strip()
            alu_wb = self.MEM_WB.get('alu_result_wb', 0); fpu_wb = self.MEM_WB.get('fpu_result_wb', 0.0)
            mem_d = self.MEM_WB.get('mem_data', 0); fp_mem_d = self.MEM_WB.get('fp_mem_data', 0.0)
            wr_r = self.MEM_WB.get('write_reg', -1); wr_f = self.MEM_WB.get('write_fp_reg', -1)
            reg_w = self.MEM_WB.get('reg_write', False); fp_reg_w = self.MEM_WB.get('fp_reg_write', False)
            mr = self.MEM_WB.get('mem_read', False); fmr = self.MEM_WB.get('fp_mem_read', False)

            details = f"PC={mem_wb_pc:<3} | {mem_wb_instr:<25}"
            wb_val_str = "N/A"
            wb_dest_str = "-"

            if reg_w and wr_r != -1:
                wb_dest_str = f"x{wr_r:<2}"
                wb_val = mem_d if mr else alu_wb
                wb_val_str = f"0x{wb_val:<8x}"
            elif fp_reg_w and wr_f != -1:
                wb_dest_str = f"f{wr_f:<2}"
                wb_val = fp_mem_d if fmr else fpu_wb
                wb_val_str = f"{wb_val:<8.3f}"

            details += f" | WB: {(wb_dest_str + ' <= ' + wb_val_str) if wb_dest_str != '-' else '-'}"
            print(f"  MEM/WB: {details}")
        print()

    def run_cycle(self):
        if DEBUG_PRINT: print(f"\n--- Start Cycle {self.clock+1} ---")
        if DEBUG_PRINT: print(f"Current PC: {self.pc}, Stall Flag: {self.stall}, Flush Flag: {self.flush}, Branch Target: {self.branch_target_pc}")
        current_state = {
            "IF_ID": self.IF_ID.copy(),
            "ID_EX": self.ID_EX.copy(),
            "EX_MEM": self.EX_MEM.copy(),
            "MEM_WB": self.MEM_WB.copy(),
            "PC": self.pc,
            "Stall": self.stall,
            "Flush": self.flush,
            "BranchTarget": self.branch_target_pc
        }

        self.wb_stage()
        self.mem_stage()
        self.ex_stage()
        self.id_stage()
        self.if_stage()

        self.clock += 1
        self.print_pipeline_state()

        if self.stall:
            if DEBUG_PRINT: print(f"Debug Cycle {self.clock}: Stall flag activated this cycle. PC remains {self.pc}. Clearing stall flag for next cycle.")
            self.stall = False
        elif self.branch_target_pc != -1:
            if DEBUG_PRINT: print(f"Debug Cycle {self.clock}: Branch/Jump taken. PC changing from {self.pc} to {self.branch_target_pc}. Flush was active.")
            self.pc = self.branch_target_pc
            self.branch_target_pc = -1
            self.flush = False
        else:
            if DEBUG_PRINT: print(f"Debug Cycle {self.clock}: No stall/branch. PC incrementing from {self.pc} to {self.pc + 4}.")
            self.pc += 4

        if DEBUG_PRINT: print(f"--- End Cycle {self.clock} --- PC for next cycle: {self.pc}")


    def wb_stage(self):
        if self.MEM_WB.get('pc', -1) == -1: return

        rd = self.MEM_WB.get('write_reg', -1)
        fd = self.MEM_WB.get('write_fp_reg', -1)
        reg_w = self.MEM_WB.get('reg_write', False)
        fp_reg_w = self.MEM_WB.get('fp_reg_write', False)
        mem_r = self.MEM_WB.get('mem_read', False)
        fp_mem_r = self.MEM_WB.get('fp_mem_read', False)

        if reg_w and rd is not None and rd > 0:
            data = self.MEM_WB.get('mem_data', 0) if mem_r else self.MEM_WB.get('alu_result_wb', 0)
            self.regs[rd] = data & UINT64_MAX
            if DEBUG_PRINT: print(f"WB: Writing x{rd} = 0x{self.regs[rd]:x}")
        elif fp_reg_w and fd is not None and fd >= 0:
            data = self.MEM_WB.get('fp_mem_data', FLOAT_ZERO) if fp_mem_r else self.MEM_WB.get('fpu_result_wb', FLOAT_ZERO)
            self.fpregs[fd] = data
            if DEBUG_PRINT: print(f"WB: Writing f{fd} = {self.fpregs[fd]:.4f}")

        if self.MEM_WB.get('op', 'NOP') != 'NOP':
            self.instructions_executed += 1

    def mem_stage(self):
        current_ex_mem = self.EX_MEM
        self.MEM_WB = current_ex_mem.copy()
        if current_ex_mem.get('pc', -1) == -1: return

        addr = current_ex_mem.get('alu_result', 0) & UINT64_MAX
        mem_w = current_ex_mem.get('mem_write', False)
        fp_mem_w = current_ex_mem.get('fp_mem_write', False)
        mem_r = current_ex_mem.get('mem_read', False)
        fp_mem_r = current_ex_mem.get('fp_mem_read', False)

        mem_data_out = 0
        fp_mem_data_out = FLOAT_ZERO

        if mem_w:
            data = current_ex_mem.get('rs2_val_mem', 0)
            self.data_mem[addr] = data & UINT64_MAX
            if DEBUG_PRINT: print(f"MEM: Write Mem[0x{addr:x}] = 0x{data & UINT64_MAX:x}")
        elif fp_mem_w:
            data_fp = current_ex_mem.get('fs2_val_mem', FLOAT_ZERO)
            data_int_bits = double_to_int64(data_fp)
            self.data_mem[addr] = data_int_bits
            if DEBUG_PRINT: print(f"MEM: Write Mem[0x{addr:x}] = {data_fp:.4f} (Bits: 0x{data_int_bits:x})")
        elif mem_r:
            mem_data_out = self.data_mem.get(addr, 0)
            if DEBUG_PRINT: print(f"MEM: Read Mem[0x{addr:x}] -> 0x{mem_data_out:x}")
        elif fp_mem_r:
            int_bits = self.data_mem.get(addr, 0)
            fp_mem_data_out = int64_to_double(int_bits)
            if DEBUG_PRINT: print(f"MEM: Read Mem[0x{addr:x}] (Bits: 0x{int_bits:x}) -> {fp_mem_data_out:.4f}")

        self.MEM_WB['mem_data'] = mem_data_out
        self.MEM_WB['fp_mem_data'] = fp_mem_data_out
        self.MEM_WB['alu_result_wb'] = current_ex_mem.get('alu_result', 0)
        self.MEM_WB['fpu_result_wb'] = current_ex_mem.get('fpu_result', FLOAT_ZERO)

    def ex_stage(self):
        current_id_ex = self.ID_EX
        ex_mem_update = current_id_ex.copy()
        if current_id_ex.get('pc', -1) == -1:
            self.EX_MEM = ex_mem_update
            return

        op = current_id_ex.get('op'); rs1_idx = current_id_ex.get('rs1'); rs2_idx = current_id_ex.get('rs2'); rs3_idx = current_id_ex.get('rs3'); rd_idx = current_id_ex.get('rd'); imm = current_id_ex.get('imm', 0); pc = current_id_ex.get('pc', 0); is_fp_dest = current_id_ex.get('is_fp_dest', False)

        op_needs_rs1_val = op not in ['JAL', 'NOP', 'Unsupported']
        op_needs_rs2_val = op in ['ADD','SUB','SLL','SLT','SLTU','XOR','SRL','SRA','OR','AND','SW','BEQ','BNE','BLT','BGE','BLTU','BGEU']
        op_needs_fs1_val = op in ['FADD.D','FSUB.D','FMUL.D','FDIV.D', 'FMIN.D', 'FMAX.D', 'FSQRT.D', 'FCVT.W.D', 'FCVT.WU.D', 'FCVT.L.D', 'FCVT.LU.D', 'FEQ.D', 'FLT.D', 'FLE.D', 'FCLASS.D', 'FSGNJ.D', 'FSGNJN.D', 'FSGNJX.D', 'FMV.X.D', 'FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D']
        op_needs_fs2_val = op in ['FADD.D','FSUB.D','FMUL.D','FDIV.D', 'FMIN.D', 'FMAX.D', 'FSD', 'FEQ.D', 'FLT.D', 'FLE.D', 'FSGNJ.D', 'FSGNJN.D', 'FSGNJX.D', 'FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D']
        op_needs_fs3_val = op in ['FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D']
        op_needs_int_rs1_val = op in ['LW', 'SW', 'JALR', 'FLD', 'FSD', 'FCVT.D.W', 'FCVT.D.WU', 'FCVT.D.L', 'FCVT.D.LU', 'FMV.D.X']

        operand1 = current_id_ex.get('rs1_val', 0)
        operand2 = current_id_ex.get('rs2_val', 0)
        foperand1 = current_id_ex.get('fs1_val', FLOAT_ZERO)
        foperand2 = current_id_ex.get('fs2_val', FLOAT_ZERO)
        foperand3 = current_id_ex.get('fs3_val', FLOAT_ZERO)

        ex_mem_fwd = self.EX_MEM
        mem_wb_fwd = self.MEM_WB

        ex_mem_rd = ex_mem_fwd.get('write_reg', -1); ex_mem_fd = ex_mem_fwd.get('write_fp_reg', -1)
        ex_mem_reg_write = ex_mem_fwd.get('reg_write', False); ex_mem_fp_reg_write = ex_mem_fwd.get('fp_reg_write', False)
        ex_mem_alu_result = ex_mem_fwd.get('alu_result', 0); ex_mem_fpu_result = ex_mem_fwd.get('fpu_result', FLOAT_ZERO)

        forward_ex_mem_rs1 = False; forward_ex_mem_rs2 = False
        forward_ex_mem_fs1 = False; forward_ex_mem_fs2 = False; forward_ex_mem_fs3 = False

        if ex_mem_reg_write and ex_mem_rd != 0:
            if rs1_idx == ex_mem_rd: operand1 = ex_mem_alu_result; forward_ex_mem_rs1 = True
            if rs2_idx == ex_mem_rd: operand2 = ex_mem_alu_result; forward_ex_mem_rs2 = True

        if ex_mem_fp_reg_write and ex_mem_fd != -1:
            if rs1_idx == ex_mem_fd: foperand1 = ex_mem_fpu_result; forward_ex_mem_fs1 = True
            if rs2_idx == ex_mem_fd: foperand2 = ex_mem_fpu_result; forward_ex_mem_fs2 = True
            if rs3_idx == ex_mem_fd: foperand3 = ex_mem_fpu_result; forward_ex_mem_fs3 = True

        mem_wb_rd = mem_wb_fwd.get('write_reg', -1); mem_wb_fd = mem_wb_fwd.get('write_fp_reg', -1)
        mem_wb_reg_write = mem_wb_fwd.get('reg_write', False); mem_wb_fp_reg_write = mem_wb_fwd.get('fp_reg_write', False)
        mem_wb_mem_read = mem_wb_fwd.get('mem_read', False); mem_wb_fp_mem_read = mem_wb_fwd.get('fp_mem_read', False)
        wb_data_int = mem_wb_fwd.get('mem_data', 0) if mem_wb_mem_read else mem_wb_fwd.get('alu_result_wb', 0)
        fp_wb_data = mem_wb_fwd.get('fp_mem_data', FLOAT_ZERO) if mem_wb_fp_mem_read else mem_wb_fwd.get('fpu_result_wb', FLOAT_ZERO)

        forward_mem_wb_rs1 = False; forward_mem_wb_rs2 = False
        forward_mem_wb_fs1 = False; forward_mem_wb_fs2 = False; forward_mem_wb_fs3 = False

        if mem_wb_reg_write and mem_wb_rd != 0:
            if rs1_idx == mem_wb_rd and not forward_ex_mem_rs1: operand1 = wb_data_int; forward_mem_wb_rs1 = True
            if rs2_idx == mem_wb_rd and not forward_ex_mem_rs2: operand2 = wb_data_int; forward_mem_wb_rs2 = True

        if mem_wb_fp_reg_write and mem_wb_fd != -1:
            if rs1_idx == mem_wb_fd and not forward_ex_mem_fs1: foperand1 = fp_wb_data; forward_mem_wb_fs1 = True
            if rs2_idx == mem_wb_fd and not forward_ex_mem_fs2: foperand2 = fp_wb_data; forward_mem_wb_fs2 = True
            if rs3_idx == mem_wb_fd and not forward_ex_mem_fs3: foperand3 = fp_wb_data; forward_mem_wb_fs3 = True

        if DEBUG_PRINT:
             fwd_msgs = []
             if forward_ex_mem_rs1: fwd_msgs.append(f"EX/MEM->rs1({rs1_idx})=0x{operand1:x}")
             if forward_ex_mem_rs2: fwd_msgs.append(f"EX/MEM->rs2({rs2_idx})=0x{operand2:x}")
             if forward_mem_wb_rs1: fwd_msgs.append(f"MEM/WB->rs1({rs1_idx})=0x{operand1:x}")
             if forward_mem_wb_rs2: fwd_msgs.append(f"MEM/WB->rs2({rs2_idx})=0x{operand2:x}")
             if forward_ex_mem_fs1: fwd_msgs.append(f"EX/MEM->fs1({rs1_idx})={foperand1:.4f}")
             if forward_ex_mem_fs2: fwd_msgs.append(f"EX/MEM->fs2({rs2_idx})={foperand2:.4f}")
             if forward_ex_mem_fs3: fwd_msgs.append(f"EX/MEM->fs3({rs3_idx})={foperand3:.4f}")
             if forward_mem_wb_fs1: fwd_msgs.append(f"MEM/WB->fs1({rs1_idx})={foperand1:.4f}")
             if forward_mem_wb_fs2: fwd_msgs.append(f"MEM/WB->fs2({rs2_idx})={foperand2:.4f}")
             if forward_mem_wb_fs3: fwd_msgs.append(f"MEM/WB->fs3({rs3_idx})={foperand3:.4f}")
             if fwd_msgs: print(f"EX (PC={pc}): Forwarding: " + ", ".join(fwd_msgs))

        alu_result = 0
        fpu_result = FLOAT_ZERO
        mem_write = False; fp_mem_write = False
        mem_read = False; fp_mem_read = False
        reg_write = False; fp_reg_write = False
        branch_taken = False
        jump_target = -1

        write_reg = rd_idx if not is_fp_dest else -1
        write_fp_reg = rd_idx if is_fp_dest else -1

        if op not in ['SW', 'FSD', 'BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU', 'NOP', 'Unsupported']:
             if is_fp_dest:
                 if write_fp_reg is not None and write_fp_reg >= 0:
                     fp_reg_write = True; reg_write = False
                 else: fp_reg_write = False; reg_write = False
             else:
                 if write_reg is not None and write_reg > 0:
                     reg_write = True; fp_reg_write = False
                 else: reg_write = False; fp_reg_write = False

        if op == 'ADDI': alu_result = operand1 + imm
        elif op == 'SLTI': alu_result = 1 if operand1 < imm else 0
        elif op == 'SLTIU': alu_result = 1 if (operand1 & UINT64_MAX) < (imm & UINT64_MAX) else 0
        elif op == 'XORI': alu_result = operand1 ^ imm
        elif op == 'ORI': alu_result = operand1 | imm
        elif op == 'ANDI': alu_result = operand1 & imm
        elif op == 'SLLI': alu_result = operand1 << (imm & 0x3F)
        elif op == 'SRLI': alu_result = (operand1 & UINT64_MAX) >> (imm & 0x3F)
        elif op == 'SRAI': alu_result = operand1 >> (imm & 0x3F)
        elif op == 'ADD': alu_result = operand1 + operand2
        elif op == 'SUB': alu_result = operand1 - operand2
        elif op == 'SLL': alu_result = operand1 << (operand2 & 0x3F)
        elif op == 'SLT': alu_result = 1 if operand1 < operand2 else 0
        elif op == 'SLTU': alu_result = 1 if (operand1 & UINT64_MAX) < (operand2 & UINT64_MAX) else 0
        elif op == 'XOR': alu_result = operand1 ^ operand2
        elif op == 'SRL': alu_result = (operand1 & UINT64_MAX) >> (operand2 & 0x3F)
        elif op == 'SRA': alu_result = operand1 >> (operand2 & 0x3F)
        elif op == 'OR': alu_result = operand1 | operand2
        elif op == 'AND': alu_result = operand1 & operand2
        elif op == 'LW': alu_result = operand1 + imm; mem_read = True
        elif op == 'SW':
             alu_result = operand1 + imm; mem_write = True; reg_write = False; fp_reg_write = False
             ex_mem_update['rs2_val_mem'] = operand2
        elif op == 'FLD':
             alu_result = operand1 + imm; fp_mem_read = True
             if fp_reg_write: reg_write = False
        elif op == 'FSD':
             alu_result = operand1 + imm; fp_mem_write = True; reg_write = False; fp_reg_write = False
             ex_mem_update['fs2_val_mem'] = foperand2
        elif op in ['BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU']:
            reg_write = False; fp_reg_write = False; branch_cond = False
            op1_unsigned = operand1 & UINT64_MAX
            op2_unsigned = operand2 & UINT64_MAX
            if op == 'BEQ': branch_cond = (operand1 == operand2)
            elif op == 'BNE': branch_cond = (operand1 != operand2)
            elif op == 'BLT': branch_cond = (operand1 < operand2)
            elif op == 'BGE': branch_cond = (operand1 >= operand2)
            elif op == 'BLTU': branch_cond = (op1_unsigned < op2_unsigned)
            elif op == 'BGEU': branch_cond = (op1_unsigned >= op2_unsigned)

            if branch_cond:
                 label = imm
                 if isinstance(label, str):
                     target_addr = self.labels.get(label, -1)
                     if target_addr != -1:
                         branch_taken = True; jump_target = target_addr; self.flush = True;
                         if DEBUG_PRINT: print(f"EX (PC={pc}): Branch {op} TAKEN to label '{label}' ({target_addr}). Flushing.")
                     else:
                         print(f"EX Error: Branch target label '{label}' not found at PC={pc}!"); branch_taken=False
                 else:
                      print(f"EX Error: Invalid branch target (not a label string) '{label}' at PC={pc}")
                      branch_taken=False
            elif DEBUG_PRINT: print(f"EX (PC={pc}): Branch {op} Condition FALSE")

        elif op == 'JAL':
            alu_result = pc + 4;
            label = imm; target_addr = -1
            if isinstance(label, str):
                target_addr = self.labels.get(label, -1)
                if target_addr != -1:
                    jump_target = target_addr; self.flush = True; branch_taken = True;
                    if DEBUG_PRINT: print(f"EX (PC={pc}): JAL to label '{label}' ({target_addr}). Flushing.")
                else:
                    print(f"EX Error: JAL target label '{label}' not found at PC={pc}!"); reg_write = False; branch_taken=False
            else:
                 print(f"EX Error: Invalid JAL target (not a label string) '{label}' at PC={pc}")
                 reg_write = False; branch_taken=False

        elif op == 'JALR':
            return_addr = pc + 4; target_addr = (operand1 + imm) & ~1;
            alu_result = return_addr
            jump_target = target_addr; self.flush = True; branch_taken = True;
            if DEBUG_PRINT: print(f"EX (PC={pc}): JALR Target Addr={target_addr}, RA=0x{return_addr:x}. Flushing.")
        elif op == 'FADD.D': fpu_result = foperand1 + foperand2
        elif op == 'FSUB.D': fpu_result = foperand1 - foperand2
        elif op == 'FMUL.D': fpu_result = foperand1 * foperand2
        elif op == 'FDIV.D':
             if foperand2 == 0.0:
                 if foperand1 == 0.0: fpu_result = float('nan')
                 else: fpu_result = math.copysign(float('inf'), foperand1 * foperand2)
             elif math.isnan(foperand1) or math.isnan(foperand2): fpu_result = float('nan')
             elif math.isinf(foperand1) and math.isinf(foperand2): fpu_result = float('nan')
             elif math.isinf(foperand1): fpu_result = math.copysign(float('inf'), foperand1*foperand2)
             elif math.isinf(foperand2): fpu_result = math.copysign(0.0, foperand1*foperand2)
             else: fpu_result = foperand1 / foperand2
        elif op == 'FSQRT.D':
             if math.isnan(foperand1): fpu_result = float('nan')
             elif foperand1 < 0: fpu_result = float('nan');
             elif foperand1 == -0.0: fpu_result = -0.0
             else: fpu_result = math.sqrt(foperand1)
        elif op == 'FMIN.D':
             if math.isnan(foperand1) and math.isnan(foperand2): fpu_result = float('nan')
             elif math.isnan(foperand1): fpu_result = foperand2
             elif math.isnan(foperand2): fpu_result = foperand1
             elif foperand1 == 0.0 and foperand2 == 0.0: fpu_result = math.copysign(0.0, foperand1 if math.copysign(1.0,foperand1)<0 else foperand2)
             else: fpu_result = min(foperand1, foperand2)
        elif op == 'FMAX.D':
             if math.isnan(foperand1) and math.isnan(foperand2): fpu_result = float('nan')
             elif math.isnan(foperand1): fpu_result = foperand2
             elif math.isnan(foperand2): fpu_result = foperand1
             elif foperand1 == 0.0 and foperand2 == 0.0: fpu_result = math.copysign(0.0, foperand1 if math.copysign(1.0,foperand1)>0 else foperand2)
             else: fpu_result = max(foperand1, foperand2)
        elif op == 'FMADD.D': fpu_result = math.fma(foperand1, foperand2, foperand3)
        elif op == 'FMSUB.D': fpu_result = math.fma(foperand1, foperand2, -foperand3)
        elif op == 'FNMADD.D': fpu_result = math.fma(-foperand1, foperand2, -foperand3)
        elif op == 'FNMSUB.D': fpu_result = math.fma(-foperand1, foperand2, foperand3)
        elif op in ['FCVT.W.D', 'FCVT.WU.D', 'FCVT.L.D', 'FCVT.LU.D']:
             try:
                 val = round_to_integer(foperand1)
                 if op == 'FCVT.W.D': alu_result = max(INT32_MIN, min(val, INT32_MAX))
                 elif op == 'FCVT.WU.D': alu_result = max(0, min(val, UINT32_MAX))
                 elif op == 'FCVT.L.D': alu_result = max(INT64_MIN, min(val, INT64_MAX))
                 elif op == 'FCVT.LU.D': alu_result = max(0, min(val, UINT64_MAX))
             except ValueError:
                 if op == 'FCVT.W.D': alu_result = INT32_MAX
                 elif op == 'FCVT.WU.D': alu_result = UINT32_MAX
                 elif op == 'FCVT.L.D': alu_result = INT64_MAX
                 elif op == 'FCVT.LU.D': alu_result = UINT64_MAX
        elif op == 'FCVT.D.W':
             val = operand1
             if val & (1 << 31): val = val - (1 << 32)
             fpu_result = float(val)
        elif op == 'FCVT.D.WU':
             fpu_result = float(operand1 & UINT32_MAX)
        elif op == 'FCVT.D.L':
             fpu_result = float(operand1)
        elif op == 'FCVT.D.LU':
             fpu_result = float(operand1 & UINT64_MAX)
        elif op == 'FMV.X.D':
             alu_result = double_to_int64(foperand1)
        elif op == 'FMV.D.X':
             fpu_result = int64_to_double(operand1)
        elif op == 'FEQ.D':
             if math.isnan(foperand1) or math.isnan(foperand2): alu_result = 0
             else: alu_result = 1 if foperand1 == foperand2 else 0
        elif op == 'FLT.D':
             if math.isnan(foperand1) or math.isnan(foperand2): alu_result = 0
             else: alu_result = 1 if foperand1 < foperand2 else 0
        elif op == 'FLE.D':
             if math.isnan(foperand1) or math.isnan(foperand2): alu_result = 0
             else: alu_result = 1 if foperand1 <= foperand2 else 0
        elif op == 'FCLASS.D':
             alu_result = classify_double(foperand1)
        elif op == 'FSGNJ.D': fpu_result = math.copysign(abs(foperand1), foperand2)
        elif op == 'FSGNJN.D': fpu_result = math.copysign(abs(foperand1), -foperand2)
        elif op == 'FSGNJX.D':
            sign_xor = math.copysign(1.0, foperand1) * math.copysign(1.0, foperand2)
            fpu_result = math.copysign(abs(foperand1), sign_xor)
        elif op == 'NOP' or op == 'Unsupported':
             reg_write = False; fp_reg_write = False

        final_reg_write = reg_write
        final_fp_reg_write = fp_reg_write
        if op in ['SW', 'FSD', 'BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU', 'NOP', 'Unsupported']:
             final_reg_write = False
             final_fp_reg_write = False
        if op in ['JAL', 'JALR'] and write_reg == 0: final_reg_write = False
        if op in ['FCVT.W.D', 'FCVT.WU.D', 'FCVT.L.D', 'FCVT.LU.D', 'FMV.X.D', 'FEQ.D', 'FLT.D', 'FLE.D', 'FCLASS.D']: final_fp_reg_write = False
        if op in ['FCVT.D.W', 'FCVT.D.WU', 'FCVT.D.L', 'FCVT.D.LU', 'FMV.D.X']: final_reg_write = False

        ex_mem_update.update({
            'alu_result': alu_result & UINT64_MAX,
            'fpu_result': fpu_result,
            'mem_write': mem_write, 'fp_mem_write': fp_mem_write,
            'mem_read': mem_read, 'fp_mem_read': fp_mem_read,
            'reg_write': final_reg_write, 'fp_reg_write': final_fp_reg_write,
            'write_reg': write_reg if final_reg_write else -1,
            'write_fp_reg': write_fp_reg if final_fp_reg_write else -1,
            'branch_taken': branch_taken
        })
        self.EX_MEM = ex_mem_update

        if branch_taken and jump_target != -1:
             self.branch_target_pc = jump_target

    def id_stage(self):
        if self.stall:
             if DEBUG_PRINT: print(f"ID: Stalled cycle detected. Injecting NOP into ID/EX.")
             self.ID_EX = NOP_INSTRUCTION.copy()
             self.ID_EX.update({'rs1_val': 0, 'rs2_val': 0, 'rs3_val': 0, 'fs1_val': FLOAT_ZERO, 'fs2_val': FLOAT_ZERO, 'fs3_val': FLOAT_ZERO})
             return

        if self.flush:
             if DEBUG_PRINT: print(f"ID: Flush signal active (from EX). Injecting NOP into ID/EX.")
             self.ID_EX = NOP_INSTRUCTION.copy()
             self.ID_EX.update({'rs1_val': 0, 'rs2_val': 0, 'rs3_val': 0, 'fs1_val': FLOAT_ZERO, 'fs2_val': FLOAT_ZERO, 'fs3_val': FLOAT_ZERO})
             return

        current_if_id = self.IF_ID.copy()
        self.ID_EX = current_if_id

        if current_if_id.get('pc', -1) == -1 or current_if_id.get('op') == 'NOP':
            self.ID_EX = NOP_INSTRUCTION.copy()
            self.ID_EX.update({'rs1_val': 0, 'rs2_val': 0, 'rs3_val': 0, 'fs1_val': FLOAT_ZERO, 'fs2_val': FLOAT_ZERO, 'fs3_val': FLOAT_ZERO})
            return

        instr = current_if_id
        op = instr.get('op')
        rs1_idx = instr.get('rs1'); rs2_idx = instr.get('rs2'); rs3_idx = instr.get('rs3')

        needs_rs1 = op not in ['JAL', 'NOP', 'Unsupported']
        needs_rs2 = op in ['ADD','SUB','SLL','SLT','SLTU','XOR','SRL','SRA','OR','AND','SW','BEQ','BNE','BLT','BGE','BLTU','BGEU']
        needs_rs3 = False
        needs_fs1 = op in ['FADD.D','FSUB.D','FMUL.D','FDIV.D', 'FMIN.D', 'FMAX.D', 'FSQRT.D', 'FCVT.W.D', 'FCVT.WU.D', 'FCVT.L.D', 'FCVT.LU.D', 'FEQ.D', 'FLT.D', 'FLE.D', 'FCLASS.D', 'FSGNJ.D', 'FSGNJN.D', 'FSGNJX.D', 'FMV.X.D', 'FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D']
        needs_fs2 = op in ['FADD.D','FSUB.D','FMUL.D','FDIV.D', 'FMIN.D', 'FMAX.D', 'FSD', 'FEQ.D', 'FLT.D', 'FLE.D', 'FSGNJ.D', 'FSGNJN.D', 'FSGNJX.D', 'FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D']
        needs_fs3 = op in ['FMADD.D', 'FMSUB.D', 'FNMSUB.D', 'FNMADD.D']
        needs_int_rs1 = op in ['LW', 'SW', 'JALR', 'FLD', 'FSD', 'FCVT.D.W', 'FCVT.D.WU', 'FCVT.D.L', 'FCVT.D.LU', 'FMV.D.X']

        rs1_val = self.regs.get(rs1_idx, 0) if (needs_rs1 or needs_int_rs1) and rs1_idx is not None and rs1_idx >= 0 else 0
        rs2_val = self.regs.get(rs2_idx, 0) if needs_rs2 and rs2_idx is not None and rs2_idx >= 0 else 0
        rs3_val = self.regs.get(rs3_idx, 0) if needs_rs3 and rs3_idx is not None and rs3_idx >= 0 else 0
        fs1_val = self.fpregs.get(rs1_idx, FLOAT_ZERO) if needs_fs1 and rs1_idx is not None and rs1_idx >= 0 else FLOAT_ZERO
        fs2_val = self.fpregs.get(rs2_idx, FLOAT_ZERO) if needs_fs2 and rs2_idx is not None and rs2_idx >= 0 else FLOAT_ZERO
        fs3_val = self.fpregs.get(rs3_idx, FLOAT_ZERO) if needs_fs3 and rs3_idx is not None and rs3_idx >= 0 else FLOAT_ZERO

        self.ID_EX.update({'rs1_val': rs1_val, 'rs2_val': rs2_val, 'rs3_val': rs3_val,
                           'fs1_val': fs1_val, 'fs2_val': fs2_val, 'fs3_val': fs3_val})

        ex_mem_instr = self.EX_MEM
        ex_op = ex_mem_instr.get('op')
        ex_is_load = ex_op in ['LW', 'FLD']

        stall_needed = False
        hazard_msg = ""

        if ex_is_load:
            ex_rd = ex_mem_instr.get('write_reg', -1)
            ex_fd = ex_mem_instr.get('write_fp_reg', -1)
            ex_reg_w = ex_mem_instr.get('reg_write', False)
            ex_fp_reg_w = ex_mem_instr.get('fp_reg_write', False)

            if ex_op == 'LW' and ex_reg_w and ex_rd != 0:
                 if ((needs_rs1 and rs1_idx == ex_rd) or
                     (needs_int_rs1 and rs1_idx == ex_rd) or
                     (needs_rs2 and rs2_idx == ex_rd)):
                     stall_needed = True
                     hazard_msg = f"Hazard: Stall LW(x{ex_rd}) in EX needed by {op} in ID."

            elif ex_op == 'FLD' and ex_fp_reg_w and ex_fd != -1:
                 if ((needs_fs1 and rs1_idx == ex_fd) or
                     (needs_fs2 and rs2_idx == ex_fd) or
                     (needs_fs3 and rs3_idx == ex_fd)):
                     stall_needed = True
                     hazard_msg = f"Hazard: Stall FLD(f{ex_fd}) in EX needed by {op} in ID."

        if stall_needed:
            if DEBUG_PRINT: print(f"ID (PC={instr.get('pc')}): {hazard_msg}")
            self.stall = True
            self.ID_EX = NOP_INSTRUCTION.copy()
            self.ID_EX.update({'rs1_val': 0, 'rs2_val': 0, 'rs3_val': 0, 'fs1_val': FLOAT_ZERO, 'fs2_val': FLOAT_ZERO, 'fs3_val': FLOAT_ZERO})

    def if_stage(self):
        if self.stall:
             if DEBUG_PRINT: print(f"IF: Stall signal active. Keeping PC={self.pc}. IF/ID register unchanged.")
             return

        if self.flush:
             if DEBUG_PRINT: print(f"IF: Flush signal active. Fetching NOP instead of PC={self.pc}.")
             self.IF_ID = NOP_INSTRUCTION.copy()
             return

        fetched = self.instr_mem.get(self.pc)
        if fetched:
            self.IF_ID = fetched.copy()
            self.IF_ID['pc'] = self.pc
            if DEBUG_PRINT: print(f"IF: Fetched PC={self.pc}: {self.IF_ID.get('raw_str', '').strip()}")
        else:
            self.IF_ID = NOP_INSTRUCTION.copy()
            max_addr = max(self.instr_mem.keys()) if self.instr_mem else -1
            if self.pc <= max_addr + 4 :
                 if DEBUG_PRINT: print(f"IF: No instruction at PC={self.pc}, inserting NOP.")

    def run(self, max_cycles=100):
        print("\n--- Instructions Loaded ---")
        addr_to_labels = collections.defaultdict(list)
        for label, addr in self.labels.items(): addr_to_labels[addr].append(label)
        sorted_instr_addrs = sorted(self.instr_mem.keys())
        for addr in sorted_instr_addrs:
             labels_at_addr = [lbl for lbl, lbl_addr in self.labels.items() if lbl_addr == addr]
             for label in labels_at_addr: print(f"{label}:")
             raw_s = self.instr_mem[addr].get('raw_str', 'ERROR: Instruction string missing')
             print(f"  {addr:<3}: {raw_s}")
        print("---------------------------\n")

        fp_val1 = 4.5; int_val1 = 1234
        self.data_mem[1000] = double_to_int64(fp_val1)
        self.data_mem[1008] = int_val1
        self.fpregs[1] = 1.5; self.fpregs[2] = -2.75; self.fpregs[3] = 10.0
        self.fpregs[4] = 0.0; self.fpregs[5] = -0.0; self.fpregs[6] = float('inf')
        self.fpregs[7] = float('nan')
        self.regs[10] = 1000; self.regs[11] = 4; self.regs[12] = -8

        print("--- Starting Simulation ---")
        pipeline_empty = False
        max_addr = max(self.instr_mem.keys()) if self.instr_mem else -1

        while self.clock < max_cycles:
            is_if_nop = self.IF_ID.get('op') == 'NOP' and self.IF_ID.get('pc') == -1
            is_id_nop = self.ID_EX.get('op') == 'NOP' and self.ID_EX.get('pc') == -1
            is_ex_nop = self.EX_MEM.get('op') == 'NOP' and self.EX_MEM.get('pc') == -1
            is_mem_nop = self.MEM_WB.get('op') == 'NOP' and self.MEM_WB.get('pc') == -1
            pc_past_code = self.pc > max_addr

            if is_if_nop and is_id_nop and is_ex_nop and is_mem_nop and pc_past_code:
                 if self.clock > 0 :
                    pipeline_empty = True
                    break
                 else: pass

            try:
                self.run_cycle()
            except Exception as e:
                print(f"\n!!!! SIMULATION ERROR at Cycle {self.clock} !!!!")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Details: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"Total Cycles: {self.clock}")
        print(f"Instructions Executed (WB): {self.instructions_executed}")

instructions = [
    "ADDI x10, x0, 1000",
    "ADDI x11, x0, 4",
    "ADDI x12, x0, -8",
    "FLD f8, 0(x10)",
    "FADD.D f9, f8, f1",
    "FSUB.D f10, f8, f1",
    "FMUL.D f11, f10, f10",
    "FSD f11, 8(x10)",
    "FMADD.D f12, f1, f8, f9",
    "FSQRT.D f13, f11",
    "FSQRT.D f14, f2",
    "ADDI x13, x0, 42",
    "ADDI x14, x0, -10",
    "FCVT.D.W f15, x13",
    "FCVT.D.W f16, x14",
    "FCVT.W.D x15, f9",
    "FCVT.W.D x16, f2",
    "FMV.D.X f17, x15",
    "FMV.X.D x17, f10",
    "FEQ.D x18, f9, f8",
    "FLT.D x19, f1, f10",
    "FLE.D x20, f10, f13",
    "FSGNJ.D f18, f1, f2",
    "FSGNJN.D f19, f1, f2",
    "FSGNJX.D f20, f1, f2",
    "FMIN.D f21, f1, f2",
    "FMAX.D f22, f1, f2",
    "FCLASS.D x21, f1",
    "FCLASS.D x22, f5",
    "FCLASS.D x23, f6",
    "FCLASS.D x24, f7",
    "ADDI x25, x0, 1",
    "BEQ x19, x25, fpu_end",
    "ADDI x1, x0, 999",
    "ADDI x1, x0, 999",
    "fpu_end:",
    "ADDI x30, x0, 123"
]

processor = Processor(instructions)
processor.run(max_cycles=80)
