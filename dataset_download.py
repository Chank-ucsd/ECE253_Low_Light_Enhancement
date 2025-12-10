import gdown
import os

id = '1I1BzIcrML9twWGw0QygcBxXmOywRhONP'

output = 'cutted_dataset.tar.gz'

gdown.download(id=id, output=output, quiet=False)
os.system(f'tar -xzvf {output}')
os.system(f'mv cut_dataset dataset')
os.remove(output)