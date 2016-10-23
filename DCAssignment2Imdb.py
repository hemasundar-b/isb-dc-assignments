from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import re

class IMDBScrape(object):
    #This is the main class with below fuctions
    # 1. __init__           : This function initialises the variables such the website links
    # 2. readUrl            : This function read the website url and return the content
    # 3. parse250Content    : This function creates a list of 250 movies and year, also writes to a file
    # 4. parseAllDetails    : This function parses the individual movie sites and get the required details
    # 5. createDataFrame    : This function reads the output file containing the details and creates the data frame

    def __init__(self):
        self.mainOutFile = "imdb_details.txt"
        self.moviesBetween1996and1998File = 'movies_between_1996_1998.txt'
        self.mainPage = 'http://www.imdb.com/'
        self.top250Url = 'http://www.imdb.com/chart/top?ref_=nv_wl_img_3'
        self.genreCountFile = 'genre_count.txt'

    def readUrl(self,paramUrl):
        content = urlopen(paramUrl).read()
        soup = BeautifulSoup(content,"lxml")
        return soup

    def parse250Content(self):
        top250List = []
        paramSoup = self.readUrl(self.top250Url)
        titleColumnsList = paramSoup.findAll("td", {"class" : "titleColumn"})
        moviesBetweenHandle = open(self.moviesBetween1996and1998File,'w')
        moviesBetweenHandle.write("MovieRank" + "\t" + "MovieName" + "\t" + "Year" + "\n")
        for var in titleColumnsList:
            href = var.find('a').get('href')
            year = var.find('span').next.replace('(','').replace(')','')
            titleColumnVar = re.sub(' +',' ',var.text.replace('\n',''))
            position = titleColumnVar.split('.')[0].strip()
            movieTitle = titleColumnVar.split('.')[1].strip().split("(")[0]
            tempList = [position,movieTitle,href]
            if year >= '1996' and year <= '1998':
                outStr = position + '\t' + movieTitle + '\t' + year
                moviesBetweenHandle.write(outStr + "\n")
            top250List.append(tempList)
        moviesBetweenHandle.close()
        print("Getting the details of Movies and Years completed")
        return top250List

    def parseAllDetails(self,param250List):
        detailsList = ['MovieName','Genres','Director','Actors','Tagline','Summary','BoxOfficeBudget','BoxOfficeGross']
        mainList = []
        mainOutputFileHandle = open(self.mainOutFile,'w')
        headerStr = "\t".join(detailsList)
        mainOutputFileHandle.write(headerStr + "\n")
        for entry in param250List:
            tempList = []
            position = entry[0]
            movieName = entry[1]
            href = entry[2]
            completeHref = self.mainPage + href
            soup = self.readUrl(completeHref)
            summaryText = soup.find("div",{"class" : "summary_text"}).text.strip()
            director = soup.find("span",{"itemprop" : "director"}).text.strip()
            actors   = soup.find("span",{"itemprop" : "actors"}).text.strip()
            genres = soup.findAll("span",{"itemprop" : "genre"})
            genreList  = []
            for i in genres:
                genreList.append(i.string)
            genreStr = "|".join(genreList)
            tempList.append(movieName)
            tempList.append(genreStr)
            tempList.append(director)
            tempList.append(actors)
            tempTag = soup.findAll("div",{"class" : "txt-block"})
            tempDict = {}
            #print(completeHref)
            for tag in tempTag:
                strTag = str(tag).replace("\n",'')
                if ("Taglines" in strTag or "Budget" in strTag or "Gross" in strTag):
                    keyValuePair = re.sub(' +',' ',str(tag.text).replace("\n",'').strip())
                    key = keyValuePair.split(":")[0]
                    val = keyValuePair.split(":")[1]
                    tempDict[key] = val
            if ("Taglines" in tempDict.keys()):
                tagVar = tempDict["Taglines"]
            else:
                tagVar = "NA"
            if("Budget" in tempDict.keys()):
                budgetVar = tempDict["Budget"]
            else:
                budgetVar = "NA"
            if ("Gross" in tempDict.keys()):
                grossVar = tempDict["Gross"]
            else:
                grossVar = "NA"
            tempList.append(tagVar)
            tempList.append(summaryText)
            tempList.append(budgetVar)
            tempList.append(grossVar)
            mainList.append(tempList)
            mainOutputFileHandle.write("\t".join(tempList) + "\n")
            print("Fetching details for Movie : " + movieName + " Completed")
        mainOutputFileHandle.close()
        return mainList

    def createDataFrame(self):
        inputLines = open("imdb_details.txt").readlines()
        stripNewLines = list(map(lambda x : x.strip('\n'),inputLines))
        listOfList = list(map(lambda x : x.split("\t"),stripNewLines))
        headers = listOfList[0]
        lol = listOfList[1:]
        df = pd.DataFrame(lol,columns=headers)
        df['MainGenre'] = list(map(lambda x : x.split("|")[0],df['Genres']))
        groupByResults = df.groupby("MainGenre").size()
        genreCountFile = 'genre_count.txt'
        groupByResults.to_csv(genreCountFile)

if __name__ == "__main__":
    scrape = IMDBScrape()
    movie250List = scrape.parse250Content()
    scrape.parseAllDetails(movie250List)
    scrape.createDataFrame()