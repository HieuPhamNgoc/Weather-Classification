import os 

class FileReader:

    def files(directory):
        files = []
        for filename in os.scandir(directory):
            path = filename.path
            path = path.replace('\\', '/')
            files.append(path)
        return files

        