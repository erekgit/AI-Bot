"""
A ParlAI agent 
Input is set using 'set_input' before calling world.parle.
"""
from parlai.core.agents import Agent
from parlai.core.message import Message


class HumanExcelAgent(Agent):

    def __init__(self, opt):
        super().__init__(opt)
        self.id = 'localExcelHuman'
        self.episodeDone = False
        self.finished = False
        self.__input = ""
        self.__conversation = []

    def set_input(self, input):
        self.__input = input or ""

    def get_conversation(self):
        return self.__conversation

    def epoch_done(self):
        return self.finished

    def observe(self, msg):
        self.__conversation.append(msg)

    def act(self):
        reply = Message()
        reply['id'] = self.getID()
        reply_text = self.__input.replace('\\n', '\n')
        reply['episode_done'] = False
        reply['text'] = reply_text
        self.__conversation.append(reply)
        return reply

    def episode_done(self):
        return self.episodeDone


"""
Facebook AI powered chatbot.

There are two functions:
  - parlai_create_world creates a world containing two agents (the AI and the human).
  - parlai_speak takes an input from the human and runs the model to get the AI response.

The entire conversation is returned by parlai_speak so it can be viewed in Excel
rather than just the last response.

See https://parl.ai/projects/blender/
"""

# -*- coding: utf-8 -*-
import discord
import re
import os
from discord.ext import commands
from discord.ext.commands import Bot
from discord.voice_client import VoiceClient
import asyncio
from flowtron import Flowtron

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task


import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import os
import argparse
import json
import sys
import numpy as np
import torch
from playsound import playsound

from flowtron import Flowtron
from torch.utils.data import DataLoader
from data import Data
from train import update_params

sys.path.insert(0, "tacotron2")
sys.path.insert(0, "tacotron2/waveglow")
from glow import WaveGlow
from scipy.io.wavfile import write



####

###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

def infer(flowtron_path, waveglow_path, text, speaker_id, n_frames, sigma,
          seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    global trainset
    global k
    global model
    global state_dict
    global speaker_vecs
    global waveglow
    

    # load waveglow
    waveglow = torch.load(waveglow_path)['model'].cuda().eval()
    waveglow.cuda().half()
    for k in waveglow.convinv:
        k.float()
    waveglow.eval()

    # load flowtron
    model = Flowtron(**model_config).cuda()
    state_dict = torch.load(flowtron_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded checkpoint '{}')" .format(flowtron_path))

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))
    speaker_vecs = trainset.get_speaker_id(speaker_id).cuda()
    speaker_vecs = speaker_vecs[None]
    

def test(text,speaker_id, n_frames, sigma,
          seed):
    
    text = trainset.get_text(text).cuda()
   
    text = text[None]

    with torch.no_grad():
        residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
        mels, attentions = model.infer(residual, speaker_vecs, text)

    for k in range(len(attentions)):
        attention = torch.cat(attentions[k]).cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        axes[0].imshow(mels[0].cpu().numpy(), origin='bottom', aspect='auto')
        axes[1].imshow(attention[:, 0].transpose(), origin='bottom', aspect='auto')
        fig.savefig('sid{}_sigma{}_attnlayer{}.png'.format(speaker_id, sigma, k))
        plt.close("all")

    audio = waveglow.infer(mels.half(), sigma=0.8).float()
    audio = audio.cpu().numpy()[0]
    # normalize audio for now
    audio = audio / np.abs(audio).max()
    print(audio.shape)
    write("sid{}_sigma{}.wav".format(speaker_id, sigma),
          data_config['sampling_rate'], audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-f', '--flowtron_path',
                        help='Path to flowtron state dict', type=str)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow state dict', type=str)
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    parser.add_argument('-i', '--id', help='Speaker id', type=int)
    parser.add_argument('-n', '--n_frames', help='Number of frames',
                        default=400, type=int)
    parser.add_argument('-o', "--output_dir", default="results/")
    parser.add_argument("-s", "--sigma", default=0.5, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)

    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    infer(args.flowtron_path, args.waveglow_path, args.text, args.id,
          args.n_frames, args.sigma, args.seed)

    print(trainset)
    #test(args.text, args.id, args.n_frames, args.sigma, args.seed)

###


def parlai_create_world(model="zoo:blender/blender_90M/model"):
    parser = ParlaiParser(True, True, 'Interactive chat with a model')
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
    )
    parser.set_defaults(interactive_mode=True, task='interactive')
    args = ['-t', 'blended_skill_talk', '-mf', model]
    opt = parser.parse_args(print_args=False, args=args)

    agent = create_agent(opt, requireModelExists=True)
    human_agent = HumanExcelAgent(opt)
    world = create_task(opt, [human_agent, agent])
    return world



def parlai_speak(world, input, limit=None):
    human, bot = world.get_agents()[-2:]
    human.set_input(input)
    world.parley()

    messages = [[x.get('id', ''), x.get('text', '')] for x in human.get_conversation()]

    if limit:
        messages = messages[-limit:]
        if len(messages) < limit:
            messages = ([['', '']] * (limit - len(messages))) + messages
    
    return messages



world = parlai_create_world()
client = discord.Client()
voice_channel = ""
vc = ""


@client.event
async def on_ready():

      print('We have logged in as {0.user}'.format(client))

      


@client.event
async def on_message(message):
    voice_channel = client.get_channel(int(""))


    try:
       vc = await voice_channel.connect()
    except:
      print("Error")
    if message.author == client.user:
        return

    if message.content.startswith(''):
        conversation = parlai_speak(world, message.content)
        for msg in conversation[-1]:          
            print(f"{id}: {msg}")
            if msg.replace("TransformerGenerator", "~") != '~':
               await message.channel.send(msg.replace("TransformerGenerator", "~").replace(" ' ", "'").replace(" ?", "?").replace(" .", ".").replace(" ,", ","))
            
               #os.system('python inference.py -c config.json -f models/flowtron_ljs.pt -w models/waveglow_256channels_v4_new.pt -t \"'+ msg +'\" -i 0')
               test(msg.replace(" ' ", "'").replace(" ?", "?").replace(" .", ".").replace(" ,", ","), args.id, args.n_frames, args.sigma, args.seed)
               
               
             
               vc.play(discord.FFmpegPCMAudio('sid0_sigma0.5.wav'))
               await asyncio.sleep(7)
               await vc.disconnect()
               
                  
              
              

client.run('')





#@inproceedings{roller2020recipes,
#  author={Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, #Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston},
#  title={Recipes for building an open-domain chatbot},
#  journal={arXiv preprint arXiv:2004.13637},
#  year={2020},
#}



