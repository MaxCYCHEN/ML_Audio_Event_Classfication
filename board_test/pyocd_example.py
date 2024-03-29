from pyocd.core.helpers import ConnectHelper
from pyocd.flash.file_programmer import FileProgrammer
from pyocd.core.memory_map import MemoryType
from pyocd.coresight.cortex_m import CortexM
#from pyocd.utility import conversion

from typing import (List)

import logging
logging.basicConfig(level=logging.INFO)

import librosa
import os
import time as tt
import struct
from pathlib import Path

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

#######################################################
# 
# This Block is for different application, 
# Please update per your dataset location. This example
# is ESC-like dataset format.
#
#######################################################
src_folder = r"C:\ESC-50-master\audio"
csv_annotation_file = r"C:\ESC-50-master\meta\esc50.csv"
file_extension =  ".wav"
used_classes = ["chainsaw","clock_tick","crackling_fire","crying_baby","dog","helicopter","rain","rooster","sea_waves","sneezing"]
test_split_portion = 0.2
want_test_num = 1
#######################################################
# 
# Please update the SRAM addresses basing on your MCU firmware.
#
#######################################################
load_data_done_addr = 0x200301cc
test_data_start_addr = 0x200343f0
Max_Ans_addr = 0x20000014

def write_float_data(target, write_data: List[float], start_addr: int):
    b_data = []
    for i in range(len(write_data)):
        # float => binary => int => write_memory_block32(), 
        # binary => write_memory_block32(), TypeError: unsupported operand type(s) for <<: 'bytes' and 'int'
        b_f = struct.pack('f', write_data[i])
        b_i = struct.unpack('i', bytes(b_f))
        b_data.append(b_i[0])
        #b_data.append(struct.pack('f', write_data[i]))
        
    target.write_memory_block32(start_addr, b_data)
    #target.write_memory_block32(start_addr, conversion.byte_list_to_u32le_list(b_data)

def most_frequent(List):
    unique, counts = np.unique(List, return_counts=True)
    index = np.argmax(counts)
    if unique[index] == -1 and len(counts) > 1: # if unknown equal other class, use the other class 
        sorted_indices = np.argsort(counts)[::-1] # sort the class count idx
        if counts[sorted_indices[0]] == counts[sorted_indices[1]]:
            return unique[counts[sorted_indices[1]]]
        
    return unique[index]

class ESC_CSV_Iterator:
    def __init__(self, csv_annotation_file, max_idx):
        self.test_esc_csv = None
        self.max_num = max_idx
        self.load_csv(csv_annotation_file)

    def __iter__(self):
        for i in range(len(self.test_esc_csv)):
            if i < self.max_num:
                filepath = Path(src_folder, self.test_esc_csv['filename'].iloc[i])
                label = self.test_esc_csv['category'].iloc[i]
                yield [filepath, label]
    def load_csv(self, csv_annotation_file):
        esc_csv = pd.read_csv(csv_annotation_file)
        esc_csv = esc_csv[esc_csv['category'].isin(used_classes)]
        train_esc_csv, self.test_esc_csv = train_test_split(esc_csv, test_size=test_split_portion,
                                                    random_state=133, stratify=esc_csv['category'])
        print("[INFO] Training set size : {} samples \n[INFO] Test set size : {} samples".format(len(train_esc_csv), len(self.test_esc_csv)))
        self.test_esc_csv['filename'] = self.test_esc_csv['filename'].astype('str')
        # Determine if we need to add file extension to file names
        add_file_extension = str(file_extension) not in self.test_esc_csv['filename'].iloc[0]
        if add_file_extension:
            self.test_esc_csv['filename'] = self.test_esc_csv['filename'] + str(file_extension)
    def get_test_data_num(self):
        return len(self.test_esc_csv)

def mcu_test(target_device):
    
    #######################################################
    # 
    # This Block is for different application, 
    # like inference data pre-process and saved csv format
    #
    #######################################################
    my_iterator = ESC_CSV_Iterator(csv_annotation_file, want_test_num)
    test_data_list = []
    test_data_idx = 0

    with ConnectHelper.session_with_chosen_probe(target_override=target_device) as session:
        board = session.board
        target = board.target
        #flash = target.memory_map.get_boot_memory()
    
        ###### Load firmware into device ######
        #FileProgrammer(session).program("I2S_Codec_PDMA_SCA_max.bin")
        
        ###### ram/rom info ######
        print("Part number:%s" % target.part_number)
        #memory_map = target.get_memory_map()
        #ram_region = memory_map.get_default_region_of_type(MemoryType.RAM)
        #rom_region = memory_map.get_boot_memory()
        #print("menory map:")
        #print(memory_map)
        #print("ram_region:")
        #print(ram_region)
        #print("rom_region(flash):")
        #print(rom_region)
        
        for ret_list in my_iterator:
            
            #######################################################
            # 
            # This Block is for different application, 
            # like inference data pre-process and saved csv format
            #
            #######################################################
            test_one_data_list = []
            test_one_data_list.append(Path(ret_list[0]).name)
            test_one_data_list.append(ret_list[1])
            
            wave, sr = librosa.load(ret_list[0], sr=16000, duration=10)
            test_wav_file_len = 16000 # single test (1 patch)
            per_wav_test_times = len(wave)//test_wav_file_len # how many patches per wav file
        
            Pred_list = []
            i = 0 # the test time count per file
            
            #######################################################
            # 
            # This main pyocd block for operate MCU
            #
            #######################################################
            
            target.resume()
            while(i < per_wav_test_times):
                
                #load_data_done_val = target.read32(load_data_done_addr)   
                #print(load_data_done_val)
                
                st = tt.time()
                
                write_float_data(target, wave[i*test_wav_file_len: (i+1)*test_wav_file_len], test_data_start_addr)
                target.flush()
                
                ed = tt.time()
                print("{} writing time: {}".format(i, (ed - st)))
                
                # Finish write mem
                target.write32(load_data_done_addr, 0x1)
                target.flush()
                
                # Wait for MCU finish inference
                tt.sleep(1.5)
                i += 1
                
                ans = target.read_memory_block32(Max_Ans_addr, 1)
                target.flush()
                # Hot fix
                if ans[0] == 4294967295:
                    Pred_list.append(-1)
                else:
                    Pred_list.append(ans[0])    
                #print("Ans: {}".format(ans[0]))
                
            #######################################################
            # 
            # This Block is for different application, 
            # like inference data pre-process and saved csv format
            #
            #######################################################
            Ans_clip = most_frequent(Pred_list)
            # filter out the unknown
            if Ans_clip == -1:
                ANS_CLASS = 'UnKnown'
            else:
                ANS_CLASS = used_classes[Ans_clip]
            
            test_data_idx+=1
            print("Finish {} data, result: {}, pred ANS: {}".format(test_data_idx, Pred_list, ANS_CLASS))
            
            test_one_data_list.append(Pred_list)
            test_one_data_list.append(ANS_CLASS)    
            test_one_data_list.append( 1 if (ret_list[1] == ANS_CLASS) else 0)
            test_data_list.append(test_one_data_list)
    
    test_data_df = pd.DataFrame(test_data_list).rename(columns={0: "test_file_name", 1:'real_ans', 2:'pred' , 3:'pred_ans' , 4:'Corretness'})
    test_data_df.to_csv("test_result.csv", sep=',', index=False, encoding='utf-8')
    
    correct_val = test_data_df['Corretness'].value_counts().get(1, 0)
    clip_accuracy = correct_val/test_data_idx
    print("Clip Test Accuracy: {}".format(clip_accuracy))     
    print("Finish All {} testdata.".format(test_data_idx))
      
if __name__ == "__main__":
    
    mcu_test("m467hjhae")        
    
    
   
    