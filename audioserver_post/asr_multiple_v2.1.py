import requests
import sys,os
import argparse

# register
r = requests.post('https://restasr.cls.ru.nl/auth/login', json={
"username": "SignOnASR",
"password": "SignOnASR2022"
}, headers = {'Content-Type': 'application/json'})
token = r.json()['data']['access_token']

# function upload and decode
def upload_decode(mywavfilename,code):
  url = 'https://restasr.cls.ru.nl/users/SignOnASR/audio'
  audio_path = mywavfilename
  audio_filename = os.path.basename(audio_path)
  
  multipart_form_data = {
      'file':(audio_filename,open(audio_path, 'rb'),'audio/wav')        
    }
  
  headers = {
    'Authorization': 'Bearer ' + token,
    }
  
  r2 = requests.post(url,headers=headers,files=multipart_form_data)
  id_audio = r2.json()['data']['filename']
  
  params = {"code": code,
  "text": "custom text",
  "keep": True
  }
  r3 = requests.post('https://restasr.cls.ru.nl/users/SignOnASR/audio/'+id_audio,
  json=params, headers = {'Authorization': "Bearer "+token})
  text = r3.json()['data']['nbest']
  
  ctm = r3.json()['data']['ctm']
  print(text)
  for i in range(0, len(ctm)):
    print("{}: {}".format(i, ctm[i]))

def create_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--code" ,type=int,help="Use codes for switching languages. 1. SignON_Dutch:code=1, 2. SignON_British_English:code=5, 3. SignON_Felmish:code=6, 4. SignON_Spanish:code=9")
    parser.add_argument("--wavlist",help="this text file contains the path of audio files")
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    language_code=args.code
    wavfiles=args.wavlist
    infile = open(wavfiles, 'r')
    WavFiles = infile.readlines()
    count = 0
    # Strips the newline character
    for line in WavFiles:
        try:
          count += 1
          print("wavfile {}: {}".format(count, line.strip()))
          upload_decode(line.strip(),language_code)
        except FileNotFoundError:
          print("wavfile {} {} Does not exist".format(count, line.strip()))