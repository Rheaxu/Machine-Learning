#!/usr/bin/python

# Author: Ruiyu Xu
# About: This code is for filtering data and get unique data grouped
#	by bump-id
# Time : Nov. 2013



def getUniqueBump():
    f_raw = open("rawData.csv",'r')
    f_unique = open("uniqueBumpData.txt","w")
    count = 0
    duplicateItem = []
    for line in f_raw:
	allText = line.split("\r")
	for item in allText:
	    l = item.split(",")
	    if l[1] not in duplicateItem:
		count += 1
		duplicateItem.append(l[1])
		newItem = l[0]+","+l[2]+","+l[3]
		f_unique.write(newItem)
		f_unique.write('\n')
    f_raw.close()
    f_unique.close()
    print count			   

def getUniqueObstacle():
    f_raw = open("rawData.csv",'r')
    f_unique = open("uniqueObstacleData.txt",'w')
    count = 0
    duplicateItem = []
    for line in f_raw:
	allText = line.split("\r")
	for item in allText:
	    l = item.split(",")
	    if l[0] not in duplicateItem:
		count += 1
		duplicateItem.append(l[0])
		newItem = l[0]+"\n"
		f_unique.write(newItem)
    f_raw.close()
    f_unique.close()
    print count

getUniqueBump()
getUniqueObstacle()
