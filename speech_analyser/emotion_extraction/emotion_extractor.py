import urllib.request
import urllib.parse
import json

def detectEmotion(text):
    data = {}
    data['algo'] = 'EmoTextANEW'
    data['i'] = text
    data['language'] = 'en'

    url_values = urllib.parse.urlencode(data)
    #print(url_values)  # The order may differ from below.

    url = 'http://senpy.cluster.gsi.dit.upm.es/api'
    full_url = url + '?' + url_values
    data = urllib.request.urlopen(full_url)

    response = json.loads(data.read().decode('utf-8'))

    emotion = response["entries"][0]["emotions"][0]["onyx:hasEmotion"][0]["onyx:hasEmotionCategory"]
    return emotion
