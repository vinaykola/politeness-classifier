'''
import facebook

token = 'CAAaL8RZCrZBC0BAATHJXJO5DJ0at8hP0s7f8wXnIQ7rQWDcfQ14w0p0CxIuZCgs0Y7ZBTP8XvZCXocwGJsPCOqYI5g9hNuIDOiXEj9mZAPm4ZC8fUq9i8v7QNRGkiQzRQ3GTZCBZBxZBdtIBjqun1DCrKviwBezSdZCYa0lHwx4Mnq9gys8loY0f6aWLUFGgZBcomYsZD'

graph = facebook.GraphAPI(token)
profile = graph.get_object("me")
friends = graph.get_connections("me", "friends")

friend_list = [friend['name'] for friend in friends['data']]

print friend_list
'''

import json
import urllib2

url = 'https://graph.facebook.com/search?q=the&type=post&limit=500&access_token='
token = 'CAAaL8RZCrZBC0BAHGTxC9dleZCgUyeLVR4DZA1yZAjMUNsmH7228nBdbSjP2ZBXR3vgzeFVfDiheSiFLOmaNw6UgJpoK96ycQFCADG0ESdD8DcQRnzhUMWrdbyNTZCOqy2ghJsJMbNwAYhRgKozVmXPFrsQyLW67PliZAiwwZCIrbjLpwMlHQvVATydBfRZAZAgnJEZD'
url += token

data = json.load(urllib2.urlopen(url))
#print data

posts = data[u'data']
f = open('output','w')
print >>f, data
f.close()

no_of_statuses = 0

f = open('statuses.txt','w')
f.close()

sep = '\n------------------------\n'

statuses = [post[u'message'] for post in posts if post['type'] == u'status']
if len(statuses) is 0:
    print 'No statuses',sep
for status in statuses:
    print status,sep
    with open('statuses.txt','a') as outfile:
        outfile.write(status.encode('utf-8'))
        outfile.write(sep)
    no_of_statuses += 1

no_of_pages = 1


while(True):
    try:
        url = data[u'paging'][u'next']
    except:
       print 'No of pages:', no_of_pages
       print 'No of statuses:', no_of_statuses
       break
    data = json.load(urllib2.urlopen(url))

    posts = data[u'data']

    statuses = [post[u'message'] for post in posts if post['type'] == u'status']
    if len(statuses) is 0:
        print 'No statuses'
    for status in statuses:
        print status
        with open('statuses.txt','a') as outfile:
            outfile.write(status.encode('utf-8'))
            outfile.write(sep)
        no_of_statuses += 1
    no_of_pages += 1
