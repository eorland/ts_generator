#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 15:24:49 2022

@author: eliorland
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import textwrap as tr

class Generator():

  def __init__(self):
    
    self.tokenizer =  GPT2Tokenizer.from_pretrained('gpt2')
    self.model = torch.load('ref_files/model.pt')

  def generate_from_prompt(self, prompt, num_attempts=1, entry_length=200, 
                           top_p=0.8, temperature=1., when_ready=True):

      '''Main generation function. Takes a prompt as input, 
         and generates an arbitrary number of descriptions 
         for that prompt. Returns list of unformatted product 
         descriptions. Can print formatted versions of each 
         description as soon as it's ready.'''

      self.model.eval()
      
      if not when_ready: # extra progress bar info
        
        prompt_split = prompt.split('>')
        product_name = prompt_split[1].split('#')[0]
        print('Working on:',product_name)

      generated_num = 0
      generated_output = []
      filter_value = -float("Inf")
      disable_cond = when_ready # on/off "switch" for progress bar
      
      with torch.no_grad():

        for entry_idx in tqdm(range(num_attempts),disable=disable_cond):

            print('\n')

            entry_finished = False
            
            # encode prompt
            generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length): # begin writing...
                
                outputs = self.model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                
                generated = torch.cat((generated, next_token), dim=1)
                
                if next_token in self.tokenizer.encode("<|endoftext|>"):
                    entry_finished = True
                
                if entry_finished: 
                    generated_num = generated_num + 1
                    output_list = list(generated.squeeze().numpy())
                    output_text = self.tokenizer.decode(output_list)
                    generated_output.append(output_text)
                    break
                
                
            if not entry_finished:
              output_list = list(generated.squeeze().numpy())
              output_text = f"{self.tokenizer.decode(output_list)}<|endoftext|>" 
              generated_output.append(output_text)    
            
            if when_ready:
              print(str(entry_idx+1) + '/' + str(num_attempts),end=' - ')
              self.display_output(output_text)
            
      if not when_ready:
        print('\n')
      
      return generated_output

  def write(self,products,entry_length=200,num_attempts=1,when_ready=True):
      
      '''Top level function which takes a list of product names,
         calls the main generation function, and returns the raw model 
         outputs. Tidy, human readable descriptions are displayed 
         automatically'''
         
      if not isinstance(products,list):
          products = [products]
      # to be a nested list - each entry is a list of 
      # descriptions for each product
      generated_descriptions = [] 
      
      for prompt in products:
        # format prompt for model input
        prompt = '<|startoftext|>' + prompt +'\n####\n'
        
        # run generator function
        x = self.generate_from_prompt(prompt, num_attempts=num_attempts,
                                      entry_length=entry_length,
                                      when_ready=when_ready)
        # add all descriptions based on prompt to main list
        generated_descriptions.append(x)
        
      if not when_ready: 
        for descriptions in generated_descriptions:
          for description in descriptions:
            print('\n')
            self.display_output(description)  
      
      return generated_descriptions

  def display_output(self,generated_text):
      
      '''Helper function which prints the model outputs in a 
        clean, human readable format'''

      wrapper = tr.TextWrapper(width=50)    
      split = generated_text.split('>')
      product_name = split[1].split("\n####\n")[0]
      description = split[1].split("\n####\n")[1].replace('<|endoftext|','')

      print(product_name + ':\n')

      word_list = wrapper.wrap(text=description)
      for element in word_list:
          print(element)

      print('\n')

      return
  
    