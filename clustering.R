#Caitlyn Ralph, Assignment 2

#Pre-process dataset

#read in KOS data, make sure working directory matches were KOS data was downloaded
data <- read.csv("KOS.csv",sep = "")

#create matrix of zeros
m = matrix(0,1000,2)

#populate m with frequency counts of each word (row number corresponds to word ID)
#column 1: how many times the word is used in all blog posts
#column 2: how many blog posts the word is used in
posts <- c()
for (row in data[,2:2]) {
  for (number in seq(1,1000)) {
    if (row == number) {
      m[number:number,1:1] = m[number:number,1:1] + data[row:row,3:3]
      if (is.element(data[row:row,1:1],posts)==FALSE) {
        posts <- c(posts,data[row:row,1:1])
        m[number:number,2:2] = m[number:number,1:1] + 1
      }
    }
  }
}

#convert to data.frame
mdf <- data.frame(matrix(unlist(m), nrow=1000, ncol=2))

#add Word ID to each word's frequency count
labels <- c(seq(1,1000))
rownames(mdf) <- labels

#find the knee to find the number of clusters
y <- matrix(0,nrow=20,ncol=1)
for (i in seq(1:20)) {
  y[i] <- sum(kmeans(mdf,i)$withinss)
}

x <- c(seq(1,20))

plot(x,y, main="Find the knee", xlab = "Number of clusters", ylab = "Withinss")

#knee = 4

#bottom-up hierarchical clustering
plot(hclust(dist(mdf),method="average"), main="Clustering for 1000 words",xlab="Word ID",ylab="Frequency count")

#doesn't give us much information -> lower to first 100 words
plot(hclust(dist(mdf[1:100,]),method="average"), main="Clustering for 100 words",xlab="Word ID",ylab="Frequency count")

#lower to first 50 words
plot(hclust(dist(mdf[1:50,]),method="average"), main="Clustering for 50 words",xlab="Word ID",ylab="Frequency count")

#k-means
library(cluster)
clusplot(mdf,kmeans(mdf,4)$cluster, color=TRUE, shade=TRUE, lines=0, labels=2, main="K-Means for 1000 words")
#first 100 words
clusplot(data.frame(mdf[1:100,]),kmeans(mdf[1:100,],2)$cluster, color=TRUE, shade=TRUE, lines=0, labels=2, main="K-Means for 100 words")

#k-mediods
#code help from https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/pam.html
clusplot(mdf,pam(mdf,4)$cluster, color=TRUE, shade=TRUE, lines=0, labels=2, main="K-Medoids for 1000 words")
#first 100 words
clusplot(data.frame(mdf[1:100,]),pam(mdf[1:100,],2)$cluster, color=TRUE, shade=TRUE, lines=0, labels=2, main="K-Medoids for 100 words")
