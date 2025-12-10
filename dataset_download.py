import gdown
import os

id = '1Xs2xKHQKxAP18sYr6xjHO3A29pAbq5Yg'

output = 'dataset.tar.gz'

gdown.download(id=id, output=output, quiet=False)
os.system(f'tar -xzvf {output}')
os.remove(output)