import tqdm as tqdm
import vk_api
import os
import urllib

path_to_save = 'Users'
#os.mkdir(path_to_save)

login, password = '+79292231513', ']5,gvug2P6$&)g37'
vk_session = vk_api.VkApi(login, password)

vk_session.auth()
vk = vk_session.get_api()

members = vk.groups.getMembers(group_id='bank_ubrr', count = 1000)

errors = list()

for idx, member in enumerate(members['items']):
    if str(member) in os.listdir(path_to_save):
        continue
    os.mkdir(os.path.join(path_to_save+'/'+str(member)))
    try:
        photos = vk.photos.get(owner_id=member, album_id='profile', offset=0, count=50)
    except Exception:
        continue
    for photo in photos['items']:
        keys = list(photo.keys())
        keys.reverse()
        for key in keys:
            if 'photo' in key:
                break
        try:
            urllib.request.urlretrieve(photo[key], path_to_save+'/'+str(member)+'/' + str(photo['id'])+'.png')
        except Exception:
            errors.append(member)
    if idx%10==0:
        print(idx)