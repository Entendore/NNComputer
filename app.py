import sys
import math
import torch
import torch.nn as nn
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QPlainTextEdit, QTableWidget, 
                               QTableWidgetItem, QComboBox, QLabel, QSplitter, QHeaderView)
from PySide6.QtCore import QTimer, Qt

# ==========================================
# 1. Optimized PyTorch Neural CPU Backend
# ==========================================

INSTR_BW = 32 

def tensor_to_uint(t):
    bw = len(t)
    bits = (t > 0.5).to(torch.int64)
    return int((bits * (1 << torch.arange(bw, dtype=torch.int64))).sum().item())

def tensor_to_int(t):
    val = tensor_to_uint(t)
    if (t > 0.5).to(torch.int64)[-1].item() == 1:
        val -= (1 << len(t))
    return val

class NeuralLogic:
    """Vectorized McCulloch-Pitts Neuron Logic Gates"""
    @staticmethod
    def step(x): return torch.heaviside(x, torch.tensor([1.0]))
    @staticmethod
    def AND(x, y): return NeuralLogic.step(x + y - 1.5)
    @staticmethod
    def OR(x, y): return NeuralLogic.step(x + y - 0.5)
    @staticmethod
    def NOT(x): return NeuralLogic.step(-x + 0.5)
    @staticmethod
    def NAND(x, y): return NeuralLogic.NOT(NeuralLogic.AND(x, y))
    @staticmethod
    def NOR(x, y): return NeuralLogic.NOT(NeuralLogic.OR(x, y))
    @staticmethod
    def XOR(x, y): return NeuralLogic.AND(NeuralLogic.OR(x, y), NeuralLogic.NOT(NeuralLogic.AND(x, y)))

class NeuralALU(nn.Module):
    def __init__(self, bw): 
        super().__init__()
        self.bit_width = bw
        
    def forward(self, a, b, op):
        if op in [0, 1, 6, 9, 10]:
            if op == 0: res = NeuralLogic.AND(a, b)
            elif op == 1: res = NeuralLogic.OR(a, b)
            elif op == 6: res = NeuralLogic.XOR(a, b)
            elif op == 9: res = NeuralLogic.NAND(a, b)
            else: res = NeuralLogic.NOR(a, b)
            z = 1.0 if torch.count_nonzero(res).item() == 0 else 0.0
            return res, z, 0, res[-1].item(), 0
            
        elif op == 4:  # SHL
            shifted_out = a[-1].item()
            res = torch.cat((torch.tensor([0.0]), a[:-1]))
            z = 1.0 if torch.count_nonzero(res).item() == 0 else 0.0
            return res, z, shifted_out, res[-1].item(), 0
            
        elif op == 5:  # SHR
            shifted_out = a[0].item()
            res = torch.cat((a[1:], torch.tensor([0.0])))
            z = 1.0 if torch.count_nonzero(res).item() == 0 else 0.0
            return res, z, shifted_out, res[-1].item(), 0
            
        elif op == 7:  # SRA
            shifted_out = a[0].item()
            sign_bit = a[-1:].clone()
            res = torch.cat((a[1:-1], sign_bit, sign_bit))
            z = 1.0 if torch.count_nonzero(res).item() == 0 else 0.0
            return res, z, shifted_out, res[-1].item(), 0
            
        elif op == 8:  # MUL
            val_a = tensor_to_int(a)
            val_b = tensor_to_int(b)
            res_val = (val_a * val_b) & ((1 << self.bit_width) - 1)
            res = torch.zeros(self.bit_width)
            for i in range(self.bit_width):
                if (res_val >> i) & 1: res[i] = 1.0
            z = 1.0 if res_val == 0 else 0.0
            return res, z, 0, res[-1].item(), 0
            
        elif op == 11: # DIV
            val_a = tensor_to_int(a)
            val_b = tensor_to_int(b)
            res_val = (val_a // val_b) & ((1 << self.bit_width) - 1) if val_b != 0 else 0
            res = torch.zeros(self.bit_width)
            for i in range(self.bit_width):
                if (res_val >> i) & 1: res[i] = 1.0
            z = 1.0 if res_val == 0 else 0.0
            v = 1.0 if val_b == 0 else 0.0
            return res, z, 0, res[-1].item(), v
            
        elif op == 12: # MOD
            val_a = tensor_to_int(a)
            val_b = tensor_to_int(b)
            res_val = (val_a % val_b) & ((1 << self.bit_width) - 1) if val_b != 0 else 0
            res = torch.zeros(self.bit_width)
            for i in range(self.bit_width):
                if (res_val >> i) & 1: res[i] = 1.0
            z = 1.0 if res_val == 0 else 0.0
            v = 1.0 if val_b == 0 else 0.0
            return res, z, 0, res[-1].item(), v
            
        elif op in [2, 3]:  # ADD/SUB
            b = NeuralLogic.NOT(b) if op == 3 else b
            carry = torch.tensor([1.0]) if op == 3 else torch.tensor([0.0])
            
            P = NeuralLogic.XOR(a, b)
            G = NeuralLogic.AND(a, b)
            
            res = torch.zeros(self.bit_width)
            carry_in_msb = 0.0
            carry_out_msb = 0.0
            
            for i in range(self.bit_width):
                if i == self.bit_width - 1:
                    carry_in_msb = carry.item()
                res[i] = NeuralLogic.XOR(P[i], carry)
                carry = NeuralLogic.OR(G[i], NeuralLogic.AND(P[i], carry))
                if i == self.bit_width - 1:
                    carry_out_msb = carry.item()
                
            z = 1.0 if torch.count_nonzero(res).item() == 0 else 0.0
            c = carry_out_msb if op == 2 else (1.0 - carry_out_msb)
            s = res[self.bit_width-1].item()
            v = 1.0 if (carry_in_msb != carry_out_msb) else 0.0
            
            return res, z, c, s, v

class NeuralCPU:
    def __init__(self, bit_width):
        self.bw = bit_width
        self.alu = NeuralALU(bit_width)
        self.rf = [torch.zeros(bit_width) for _ in range(8)]
        self.ram_size = 32 if bit_width == 16 else 64
        self.ram = [torch.zeros(bit_width) for _ in range(self.ram_size)]
        
        self.pc = torch.zeros(bit_width)
        self.sp = torch.zeros(bit_width)
        
        sp_val = self.ram_size - 1
        for i in range(bit_width):
            if (sp_val >> i) & 1:
                self.sp[i] = 1.0
        
        self.flags = [0, 0, 0, 0]
        self.rom = []
        self.cycles = 0
        self.output_port = 0
        
        # Maxed out 6-bit ISA (64 Opcodes perfectly packed)
        self.OPCODES = {
            "AND": 0, "OR": 1, "ADD": 2, "SUB": 3, "SHL": 4, "SHR": 5, "XOR": 6, 
            "SRA": 7, "MUL": 8, "NAND": 9, "NOR": 10, "DIV": 11, "MOD": 12,
            "ADDI": 13, "ANDI": 14, "ORI": 15, "XORI": 16, "LI": 17,
            "LSLI": 18, "LSRI": 19, "SUBI": 20, "CMPI": 21,
            "LOAD": 22, "STORE": 23, "LOADR": 24, "STORER": 25, "LOADX": 26, "STOREX": 27,
            "JMP": 28, "BEQ": 29, "BNE": 30, "BLT": 31, "BGT": 32, "BLE": 33, "BGE": 34,
            "PUSH": 35, "POP": 36, "CALL": 37, "RET": 38,
            "CMP": 39, "MOV": 40, "INC": 41, "DEC": 42, "SWAP": 43, "NEG": 44,
            "ROL": 45, "ROR": 46, "BTST": 47, "BSET": 48, "BCLR": 49,
            "OUT": 50,
            "MIN": 51, "MAX": 52, "ABS": 53, "POPCNT": 54, "REV": 55, "SEXT": 56,
            "MOVZ": 57, "MOVNZ": 58, "ADDSP": 59,
            "LUI": 60, "EXTB": 61, "CLZ": 62,
            "HLT": 63
        }
        self.OPCODES_REV = {v: k for k, v in self.OPCODES.items()}

    def safe_addr(self, val):
        return max(0, min(val, self.ram_size - 1))

    def int_to_tensor(self, val):
        val = val & ((1 << self.bw) - 1)
        t = torch.zeros(self.bw)
        for i in range(self.bw):
            if (val >> i) & 1: t[i] = 1.0
        return t

    def load_rom(self, program): 
        self.rom = program

    def step(self):
        pc_int = tensor_to_uint(self.pc)
        if pc_int >= len(self.rom): 
            return False, "HALT"
        
        instr = self.rom[pc_int]
        
        op_int = int((instr[0:6] * torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.float32)).sum().item())
        dest = int((instr[6:9] * torch.tensor([1, 2, 4], dtype=torch.float32)).sum().item())
        src1 = int((instr[9:12] * torch.tensor([1, 2, 4], dtype=torch.float32)).sum().item())
        imm = int((instr[12:32] * (1 << torch.arange(20, dtype=torch.int64))).sum().item())
        if imm & (1 << 19): imm -= (1 << 20)
        
        op_name = self.OPCODES_REV.get(op_int, "HLT")
            
        one_t = torch.zeros(self.bw)
        one_t[0] = 1.0
        next_pc = self.alu(self.pc, one_t, 2)[0]
        log_msg = ""

        if op_name == "HLT":
            return False, "HLT"
            
        elif op_name in ["AND", "OR", "ADD", "SUB", "SHL", "SHR", "XOR", "SRA", "MUL", "NAND", "NOR", "DIV", "MOD"]:
            res, z, c, s, v = self.alu(self.rf[dest], self.rf[src1], op_int)
            if dest != 0: self.rf[dest] = res
            self.flags = [z, c, s, v]
            log_msg = f"{op_name} R{dest}, R{src1}"
            
        elif op_name in ["ADDI", "ANDI", "ORI", "XORI", "SUBI", "CMPI"]:
            imm_t = self.int_to_tensor(imm)
            alu_op = 2 if op_name in ["ADDI", "SUBI", "CMPI"] else (0 if op_name == "ANDI" else (1 if op_name == "ORI" else 6))
            if op_name in ["SUBI", "CMPI"]: alu_op = 3
                
            res, z, c, s, v = self.alu(self.rf[src1], imm_t, alu_op)
            if op_name != "CMPI":
                if dest != 0: self.rf[dest] = res
            self.flags = [z, c, s, v]
            log_msg = f"{op_name} R{dest}, R{src1}, {imm}"
            
        elif op_name in ["LSLI", "LSRI"]:
            shift_amt = max(0, min(imm, self.bw - 1))
            res = self.rf[src1].clone()
            for _ in range(shift_amt):
                if op_name == "LSLI": res = torch.cat((torch.tensor([0.0]), res[:-1]))
                else: res = torch.cat((res[1:], torch.tensor([0.0])))
            z = 1.0 if torch.count_nonzero(res).item() == 0 else 0.0
            if dest != 0: self.rf[dest] = res
            self.flags = [z, 0, res[-1].item(), 0]
            log_msg = f"{op_name} R{dest}, R{src1}, {shift_amt}"
            
        elif op_name in ["ROL", "ROR"]:
            shift_amt = max(0, min(imm, self.bw - 1))
            shifts = shift_amt if op_name == "ROL" else -shift_amt
            res = torch.roll(self.rf[src1], shifts=shifts)
            z = 1.0 if torch.count_nonzero(res).item() == 0 else 0.0
            if dest != 0: self.rf[dest] = res
            self.flags = [z, 0, res[-1].item(), 0]
            log_msg = f"{op_name} R{dest}, R{src1}, {shift_amt}"

        elif op_name == "MIN":
            val_a = tensor_to_int(self.rf[dest])
            val_b = tensor_to_int(self.rf[src1])
            res = self.int_to_tensor(min(val_a, val_b))
            if dest != 0: self.rf[dest] = res
            log_msg = f"MIN R{dest}, R{src1}"
            
        elif op_name == "MAX":
            val_a = tensor_to_int(self.rf[dest])
            val_b = tensor_to_int(self.rf[src1])
            res = self.int_to_tensor(max(val_a, val_b))
            if dest != 0: self.rf[dest] = res
            log_msg = f"MAX R{dest}, R{src1}"
            
        elif op_name == "ABS":
            val = tensor_to_int(self.rf[src1])
            res = self.int_to_tensor(abs(val))
            if dest != 0: self.rf[dest] = res
            log_msg = f"ABS R{dest}, R{src1}"
            
        elif op_name == "POPCNT":
            count = int(torch.count_nonzero(self.rf[src1] > 0.5).item())
            res = self.int_to_tensor(count)
            if dest != 0: self.rf[dest] = res
            log_msg = f"POPCNT R{dest}, R{src1} (Count={count})"
            
        elif op_name == "REV":
            res = torch.flip(self.rf[src1], dims=[0])
            if dest != 0: self.rf[dest] = res
            log_msg = f"REV R{dest}, R{src1}"
            
        elif op_name == "SEXT":
            bits = max(1, min(imm, self.bw))
            lower_bits = self.rf[src1][:bits].clone()
            sign_bit = lower_bits[-1].item()
            res = lower_bits
            if sign_bit == 1.0:
                res = torch.cat((res, torch.ones(self.bw - bits)))
            else:
                res = torch.cat((res, torch.zeros(self.bw - bits)))
            if dest != 0: self.rf[dest] = res
            log_msg = f"SEXT R{dest}, R{src1}, {bits}"
            
        elif op_name == "MOVZ":
            if self.flags[0] == 1.0:
                if dest != 0: self.rf[dest] = self.rf[src1].clone()
                log_msg = f"MOVZ R{dest}, R{src1} (Moved)"
            else:
                log_msg = f"MOVZ R{dest}, R{src1} (Skipped)"
                
        elif op_name == "MOVNZ":
            if self.flags[0] == 0.0:
                if dest != 0: self.rf[dest] = self.rf[src1].clone()
                log_msg = f"MOVNZ R{dest}, R{src1} (Moved)"
            else:
                log_msg = f"MOVNZ R{dest}, R{src1} (Skipped)"
                
        elif op_name == "ADDSP":
            self.sp = self.alu(self.sp, self.int_to_tensor(imm), 2)[0]
            log_msg = f"ADDSP {imm}"
            
        elif op_name == "LUI":  # Load Upper Immediate
            shift = max(0, self.bw - 20)
            val = imm << shift
            if dest != 0: self.rf[dest] = self.int_to_tensor(val)
            log_msg = f"LUI R{dest}, {imm} (Val={val})"
            
        elif op_name == "EXTB":  # Extract Byte
            byte_idx = max(0, min(imm, (self.bw // 8) - 1))
            shift = byte_idx * 8
            val = (tensor_to_uint(self.rf[src1]) >> shift) & 0xFF
            if dest != 0: self.rf[dest] = self.int_to_tensor(val)
            log_msg = f"EXTB R{dest}, R{src1}, {byte_idx} (Val={val})"
            
        elif op_name == "CLZ":  # Count Leading Zeros
            val = tensor_to_uint(self.rf[src1])
            count = 0
            for i in range(self.bw - 1, -1, -1):
                if (val >> i) & 1: break
                count += 1
            if dest != 0: self.rf[dest] = self.int_to_tensor(count)
            log_msg = f"CLZ R{dest}, R{src1} (Count={count})"
            
        elif op_name == "BTST":
            bit_idx = max(0, min(imm, self.bw - 1))
            bit_val = self.rf[dest][bit_idx].item()
            self.flags[0] = 0.0 if bit_val == 1.0 else 1.0
            log_msg = f"BTST R{dest}, {bit_idx} (Set: {int(bit_val)})"
            
        elif op_name == "BSET":
            bit_idx = max(0, min(imm, self.bw - 1))
            if dest != 0: self.rf[dest][bit_idx] = 1.0
            log_msg = f"BSET R{dest}, {bit_idx}"
            
        elif op_name == "BCLR":
            bit_idx = max(0, min(imm, self.bw - 1))
            if dest != 0: self.rf[dest][bit_idx] = 0.0
            log_msg = f"BCLR R{dest}, {bit_idx}"
            
        elif op_name == "INC":
            res, z, c, s, v = self.alu(self.rf[dest], one_t, 2)
            if dest != 0: self.rf[dest] = res
            self.flags = [z, c, s, v]
            log_msg = f"INC R{dest}"
            
        elif op_name == "DEC":
            res, z, c, s, v = self.alu(self.rf[dest], one_t, 3)
            if dest != 0: self.rf[dest] = res
            self.flags = [z, c, s, v]
            log_msg = f"DEC R{dest}"
            
        elif op_name == "SWAP":
            self.rf[dest], self.rf[src1] = self.rf[src1].clone(), self.rf[dest].clone()
            log_msg = f"SWAP R{dest}, R{src1}"
            
        elif op_name == "CMP":
            res, z, c, s, v = self.alu(self.rf[dest], self.rf[src1], 3)
            self.flags = [z, c, s, v]
            log_msg = f"CMP R{dest}, R{src1}"
            
        elif op_name == "MOV":
            if dest != 0: self.rf[dest] = self.rf[src1].clone()
            log_msg = f"MOV R{dest}, R{src1}"
            
        elif op_name == "LI":
            if dest != 0: self.rf[dest] = self.int_to_tensor(imm)
            log_msg = f"LI R{dest}, {imm}"
            
        elif op_name == "NEG":
            res, z, c, s, v = self.alu(torch.zeros(self.bw), self.rf[src1], 3)
            if dest != 0: self.rf[dest] = res
            self.flags = [z, c, s, v]
            log_msg = f"NEG R{dest}, R{src1}"
            
        elif op_name == "LOAD": 
            self.rf[dest] = self.ram[self.safe_addr(imm)]
            log_msg = f"LOAD R{dest}, [{imm}]"
        elif op_name == "STORE": 
            self.ram[self.safe_addr(imm)] = self.rf[src1]
            log_msg = f"STORE [{imm}], R{src1}"
        elif op_name == "LOADR": 
            addr = self.safe_addr(tensor_to_uint(self.rf[src1]))
            self.rf[dest] = self.ram[addr]
            log_msg = f"LOADR R{dest}, [R{src1}]"
        elif op_name == "STORER": 
            addr = self.safe_addr(tensor_to_uint(self.rf[dest]))
            self.ram[addr] = self.rf[src1]
            log_msg = f"STORER [R{dest}], R{src1}"
        elif op_name == "LOADX":
            base_addr = tensor_to_int(self.rf[src1])
            addr = self.safe_addr(base_addr + imm)
            self.rf[dest] = self.ram[addr]
            log_msg = f"LOADX R{dest}, R{src1}, {imm} (Addr={addr})"
        elif op_name == "STOREX":
            base_addr = tensor_to_int(self.rf[dest])
            addr = self.safe_addr(base_addr + imm)
            self.ram[addr] = self.rf[src1]
            log_msg = f"STOREX R{dest}, R{src1}, {imm} (Addr={addr})"
            
        elif op_name == "JMP": 
            next_pc = self.int_to_tensor(imm)
            log_msg = f"JMP {imm}"
        elif op_name == "BEQ" and self.flags[0] == 1.0: 
            next_pc = self.int_to_tensor(imm)
            log_msg = f"BEQ {imm} (Taken)"
        elif op_name == "BEQ": log_msg = f"BEQ {imm} (Skipped)"
        elif op_name == "BNE" and self.flags[0] == 0.0: 
            next_pc = self.int_to_tensor(imm)
            log_msg = f"BNE {imm} (Taken)"
        elif op_name == "BNE": log_msg = f"BNE {imm} (Skipped)"
        elif op_name == "BLT" and (self.flags[2] != self.flags[3]):
            next_pc = self.int_to_tensor(imm)
            log_msg = f"BLT {imm} (Taken)"
        elif op_name == "BLT": log_msg = f"BLT {imm} (Skipped)"
        elif op_name == "BGT" and self.flags[0] == 0.0 and (self.flags[2] == self.flags[3]):
            next_pc = self.int_to_tensor(imm)
            log_msg = f"BGT {imm} (Taken)"
        elif op_name == "BGT": log_msg = f"BGT {imm} (Skipped)"
        elif op_name == "BLE" and (self.flags[0] == 1.0 or (self.flags[2] != self.flags[3])):
            next_pc = self.int_to_tensor(imm)
            log_msg = f"BLE {imm} (Taken)"
        elif op_name == "BLE": log_msg = f"BLE {imm} (Skipped)"
        elif op_name == "BGE" and (self.flags[0] == 1.0 or (self.flags[2] == self.flags[3])):
            next_pc = self.int_to_tensor(imm)
            log_msg = f"BGE {imm} (Taken)"
        elif op_name == "BGE": log_msg = f"BGE {imm} (Skipped)"
            
        elif op_name == "PUSH":
            self.ram[self.safe_addr(tensor_to_int(self.sp))] = self.rf[src1]
            self.sp = self.alu(self.sp, one_t, 3)[0]
            log_msg = f"PUSH R{src1}"
        elif op_name == "POP":
            self.sp = self.alu(self.sp, one_t, 2)[0]
            self.rf[dest] = self.ram[self.safe_addr(tensor_to_int(self.sp))]
            log_msg = f"POP R{dest}"
        elif op_name == "CALL":
            self.ram[self.safe_addr(tensor_to_int(self.sp))] = next_pc.clone()
            self.sp = self.alu(self.sp, one_t, 3)[0]
            next_pc = self.int_to_tensor(imm)
            log_msg = f"CALL {imm}"
        elif op_name == "RET":
            self.sp = self.alu(self.sp, one_t, 2)[0]
            next_pc = self.ram[self.safe_addr(tensor_to_int(self.sp))]
            log_msg = "RET"
        elif op_name == "OUT":
            self.output_port = tensor_to_int(self.rf[src1])
            log_msg = f"OUT R{src1} (Port={self.output_port})"
            
        self.pc = next_pc
        self.cycles += 1
        return True, log_msg

# ==========================================
# 2. Enhanced Two-Pass Assembler
# ==========================================
def assemble(code, bw):
    lines = [l.split('//')[0].strip() for l in code.split('\n') if l.strip()]
    labels = {}
    program = []
    
    for line in lines:
        if ':' in line:
            lbl, instr = line.split(':', 1)
            labels[lbl.strip()] = len(program)
            if instr.strip(): program.append(instr.strip())
        else:
            program.append(line)
        
    rom = []
    cpu_ref = NeuralCPU(bw)
    
    def parse_imm(tok):
        if tok.startswith('0x') or tok.startswith('-0x'):
            v = int(tok, 16)
        elif tok.lstrip('-').isdigit():
            v = int(tok)
        else:
            return labels.get(tok, 0)
        mask = (1 << 20) - 1
        return v & mask
        
    def clean_token(tok):
        tok = tok.strip().strip('[]').strip()
        if tok.upper().startswith('R'): tok = tok[1:]
        return tok
        
    for line in program:
        parts = line.replace(',', ' ').split()
        op = parts[0].upper()
        instr = torch.zeros(INSTR_BW)
        
        if op not in cpu_ref.OPCODES:
            raise ValueError(f"Unknown opcode: {op}")
            
        op_int = cpu_ref.OPCODES.get(op, 63)
        for i in range(6): instr[i] = (op_int >> i) & 1
            
        args = parts[1:]
        
        if op in ["RET", "HLT"]:
            pass
        elif op == "OUT":
            if len(args) > 0: 
                src1 = int(clean_token(args[0]))
                for i in range(3): instr[9+i] = (src1 >> i) & 1
        elif op in ["CALL", "JMP", "BEQ", "BNE", "BLT", "BGT", "BLE", "BGE", "LOAD", "STORE", "ADDSP"]:
            if len(args) > 0:
                val = parse_imm(args[0])
                for i in range(20): instr[12+i] = (val >> i) & 1
        elif op in ["LUI", "BTST", "BSET", "BCLR", "ROL", "ROR", "SEXT"]:
            if len(args) > 0:
                dest = int(clean_token(args[0]))
                for i in range(3): instr[6+i] = (dest >> i) & 1
            if len(args) > 1:
                val = parse_imm(args[1])
                for i in range(20): instr[12+i] = (val >> i) & 1
        elif op in ["CLZ", "ABS"]:
            if len(args) > 0:
                dest = int(clean_token(args[0]))
                for i in range(3): instr[6+i] = (dest >> i) & 1
            if len(args) > 1:
                src1 = int(clean_token(args[1]))
                for i in range(3): instr[9+i] = (src1 >> i) & 1
        elif op in ["INC", "DEC"]:
            if len(args) > 0:
                dest = int(clean_token(args[0]))
                for i in range(3): instr[6+i] = (dest >> i) & 1
        else:
            # Handle 1-operand vs 2-operand vs 3-operand instructions implicitly
            if len(args) > 0:
                dest = int(clean_token(args[0]))
                for i in range(3): instr[6+i] = (dest >> i) & 1
                src1 = dest  # Implicit src1 for 1-operand instructions
            if len(args) > 1:
                src1 = int(clean_token(args[1]))
                for i in range(3): instr[9+i] = (src1 >> i) & 1
            if len(args) > 2:
                val = parse_imm(args[2])
                for i in range(20): instr[12+i] = (val >> i) & 1
        rom.append(instr)
    return rom

# ==========================================
# 3. PySide6 GUI Frontend
# ==========================================
class NeuralCPUWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural CPU IDE - 64-Opcode Complete ISA & Hex Support")
        self.resize(1100, 750)
        self.bw = 16
        self.cpu = NeuralCPU(self.bw)
        self._last_rom = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step_cpu)
        self.init_ui()
        self.load_default_code()
        self.change_arch()

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal); main_layout.addWidget(splitter)

        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget)
        ctrl_layout = QHBoxLayout()
        self.arch_combo = QComboBox(); self.arch_combo.addItems(["16-bit", "32-bit", "64-bit"])
        self.arch_combo.setCurrentText("16-bit"); self.arch_combo.currentTextChanged.connect(self.change_arch)
        
        self.btn_asm = QPushButton("Assemble & Load"); self.btn_asm.clicked.connect(self.assemble_code)
        self.btn_step = QPushButton("Step"); self.btn_step.clicked.connect(self.step_cpu)
        self.btn_run = QPushButton("Run"); self.btn_run.clicked.connect(self.toggle_run)
        self.btn_reset = QPushButton("Reset"); self.btn_reset.clicked.connect(self.reset_cpu)

        ctrl_layout.addWidget(QLabel("Arch:")); ctrl_layout.addWidget(self.arch_combo)
        ctrl_layout.addWidget(self.btn_asm); ctrl_layout.addWidget(self.btn_step)
        ctrl_layout.addWidget(self.btn_run); ctrl_layout.addWidget(self.btn_reset)
        
        self.code_editor = QPlainTextEdit()
        self.code_editor.setStyleSheet("font-family: monospace; font-size: 14px;")
        left_layout.addLayout(ctrl_layout); left_layout.addWidget(self.code_editor)

        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget)
        self.reg_table = QTableWidget(12, 2)
        self.reg_table.setHorizontalHeaderLabels(["Register", "Value"])
        self.reg_table.verticalHeader().setVisible(False)
        self.reg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.ram_table = QTableWidget(16, 2)
        self.ram_table.setHorizontalHeaderLabels(["RAM Addr", "Value"])
        self.ram_table.verticalHeader().setVisible(False)
        self.ram_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.log_console = QPlainTextEdit(); self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: black; color: #00FF00; font-family: monospace;")

        right_layout.addWidget(QLabel("CPU Registers, PC, SP & Flags")); right_layout.addWidget(self.reg_table)
        right_layout.addWidget(QLabel("Memory (RAM) - Stack & Heap (Follows SP)")); right_layout.addWidget(self.ram_table)
        right_layout.addWidget(QLabel("Execution Log & Hardware Output")); right_layout.addWidget(self.log_console)

        splitter.addWidget(left_widget); splitter.addWidget(right_widget)
        splitter.setSizes([450, 650])

    def load_default_code(self):
        code = """// Neural CPU - Byte Extraction & Hex Demo
// Load 0xABCD into R1
LI R1, 0xABCD

// Extract upper byte (0xAB)
EXTB R2, R1, 1   // R2 = (R1 >> 8) & 0xFF
OUT R2           // Output 171 (0xAB)

// Extract lower byte (0xCD)
EXTB R3, R1, 0   // R3 = (R1 >> 0) & 0xFF
OUT R3           // Output 205 (0xCD)

// Count leading zeros in 0xAB (10101011)
// In 16-bit, it has 8 leading zeros
CLZ R4, R2       // R4 = 8
OUT R4           // Output 8

// To construct 0xABCD from bytes:
LI R5, 0xAB
LSLI R5, R5, 8   // R5 = 0xAB00
ORI R5, R5, 0xCD // R5 = 0xABCD
OUT R5           // Output 43981 (0xABCD)

HLT
"""
        self.code_editor.setPlainText(code)

    def change_arch(self):
        self.bw = int(self.arch_combo.currentText().split('-')[0])
        self.reset_cpu()

    def assemble_code(self):
        try:
            rom = assemble(self.code_editor.toPlainText(), self.bw)
            self._last_rom = rom
            self.cpu = NeuralCPU(self.bw); self.cpu.load_rom(rom)
            self.log_console.appendPlainText(f"[ASM] Assembled {len(rom)} instructions ({self.bw}-bit data, 32-bit ISA).")
            self.update_ui()
        except Exception as e:
            self.log_console.appendPlainText(f"[ERROR] {str(e)}")

    def step_cpu(self):
        if not self.cpu.rom:
            self.log_console.appendPlainText("[ERROR] No program loaded.")
            return
        pc_before = tensor_to_uint(self.cpu.pc)
        running, log_msg = self.cpu.step()
        self.log_console.appendPlainText(f"Cyc:{self.cpu.cycles} PC:{pc_before} > {log_msg}")
        if "OUT" in log_msg: 
            self.log_console.appendPlainText(f"   [HW_OUT] Port value: {self.cpu.output_port}")
        self.update_ui()
        if not running:
            self.timer.stop(); self.btn_run.setText("Run")
            self.log_console.appendPlainText("[HALT] Program finished.")

    def toggle_run(self):
        if self.timer.isActive(): 
            self.timer.stop(); self.btn_run.setText("Run")
        else: 
            self.timer.start(50); self.btn_run.setText("Pause")

    def reset_cpu(self):
        self.timer.stop(); self.btn_run.setText("Run")
        self.cpu = NeuralCPU(self.bw)
        if self._last_rom is not None:
            self.cpu.load_rom(self._last_rom)
        self.log_console.appendPlainText(f"[RESET] CPU initialized to {self.bw}-bit.")
        self.update_ui()

    def update_ui(self):
        reg_names = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "PC", "SP", "Flags(ZCSV)", "Cycles"]
        vals = [tensor_to_int(r) for r in self.cpu.rf]
        vals.append(tensor_to_uint(self.cpu.pc))
        vals.append(tensor_to_uint(self.cpu.sp))
        vals.append(f"Z:{int(self.cpu.flags[0])} C:{int(self.cpu.flags[1])} S:{int(self.cpu.flags[2])} V:{int(self.cpu.flags[3])}")
        vals.append(self.cpu.cycles)
        
        self.reg_table.setRowCount(12)
        for i, (name, val) in enumerate(zip(reg_names, vals)):
            self.reg_table.setItem(i, 0, QTableWidgetItem(name))
            self.reg_table.setItem(i, 1, QTableWidgetItem(str(val)))
            
        sp = tensor_to_uint(self.cpu.sp)
        start = max(0, min(sp - 8, self.cpu.ram_size - 16))
        self.ram_table.setRowCount(16)
        for i in range(16):
            addr = start + i
            marker = " <-- SP" if addr == sp else ""
            self.ram_table.setItem(i, 0, QTableWidgetItem(f"{addr}{marker}"))
            self.ram_table.setItem(i, 1, QTableWidgetItem(str(tensor_to_int(self.cpu.ram[addr]))))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = NeuralCPUWindow()
    window.show()
    sys.exit(app.exec())