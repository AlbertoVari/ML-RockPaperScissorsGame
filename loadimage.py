import wget
url1 = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip'
output_directory1 = '/tmp/rps.zip'
url2 = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip'
output_directory2 = '/tmp/rps-test-set.zip'
filename1 = wget.download(url1, out=output_directory1)
filename1
filename2 = wget.download(url2, out=output_directory2)
filename2
