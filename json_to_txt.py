import json

path1 = "./news_2020_ver1/NIRW2000000001.json"

with open(path1,'r') as f:
    json_data = json.load(f)

txt_file=open("news_2020_ver1.txt",'w')

txt_2 = open("news_2020_ver1_2.txt",'w')

#디렉토리 내의 json 파일만 인식해서 처리

n = 0
#l = len(json_data["document"])
#article_len =
sents = {}
#print(json_data["document"][0]["paragraph"])
for i in json_data["document"]:
    for j in i["paragraph"]:
        if n not in sents:
            sents[n] = []
            sents[n] = str(j["form"] + "\n")
        else :
            sents[n] = str(sents[n] + j["form"] + "\n")
    sents[n] = str(sents[n]+ "\n\n")
    n += 1

for i in sents:
    txt_2.write(sents[i])

txt_file.close()
txt_2.close()

