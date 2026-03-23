# Comments in R
# Declare a simple variable in R
# Any thing using 
a <- 10
b = 20
a
b
# Performing Basic Analysis 
c=a+b # the value of a and b is added to c 
c
c=a*b
c
c=a-b
c
c=a/b
d=a*b+c    
d
# Create Array 
a1=c(1,2,3,4,5,6,7,8,9,10)
a1
a2=c(1,2,3,4,5,6,7,8,9,10)
a4=c("MCA", "MBA", "BBA", "BCA")
a3=a1+a2
a3
a4
a=10
a=20
b=a
# Steps to import dataset 
# Manual Method 
mydataset=read.csv(file.choose())
mydataset
# Similarly we can import other format like .csv
dataset12=read.csv(file.choose())
#scatter Plot
View(dataset12)
# plot(dataset12$Height, dataset12$Weight)
plot(dataset12$Height, dataset12$Weight)
plot(dataset12$Height, dataset12$Weight, main="This is a relation between Height and Weight")
plot(dataset12$Height, dataset12$Weight, main="This is a relation between Height and Weight", xlab="Weight", ylab="Height")
plot(dataset12$Height, dataset12$Weight, main="This is a relation between Height and Weight", xlab="Weight", ylab="Height", col="#00ff00", pch=20)
# PCH has 0 to 25
plot(dataset12$Height, dataset12$Weight, main="This is a relation between Height and Weight", xlab="Weight", ylab="Height", col="red", pch=15, cex=2)
# Use cex 1 is default, while 0.5 means 50% smaller, and 2 means 100% larger
barplot(dataset12$Weight)
barplot(dataset12$Weight)
# Horizontal 
barplot(dataset12$Weight, horiz = TRUE)# TRUE, R is casesensitive language
# Barplot for categorical data
barplot((table(dataset12$Gender))/nrow(dataset12))
barplot(table(dataset12$Department)/nrow(dataset12))
# Multiple Scatter Plot
pairs(~dataset12$Height+dataset12$Weight+dataset12$Marks, dataset12)
pairs(~dataset12$Height+dataset12$Weight+dataset12$Marks,dataset12)
# Pie Chart
pie(table(dataset12$Department, dataset12$City))
pie(table(dataset12$Department), dataset12$City)
stem(dataset12$Marks)
#Plot 
plot(dataset12$Weight, dataset12$Marks, type="s")
plot(dataset12$Weight, dataset12$Marks, type="l")
plot(dataset12$Weight, dataset12$Marks, type="o")
plot(dataset12$Weight, dataset12$Marks, type="h")
# Advance Plotting using GGPLOT 2
library(ggplot2)
#Simple Graph
attach(dataset12)
ggplot(data=dataset12, aes(x=Gender, y=Weight))+geom_point()
ggplot(data =dataset12, aes(x=Gender, y=Weight))+geom_point()
# Simple Line Graph
ggplot(dataset12, aes(Gender, Weight))+geom_point()+geom_l
# Increase the size
ggplot(dataset12, aes(Gender, Weight))+geom_point(size=1)+geom_line()    
# Box Plot
ggplot(dataset12, aes(Weight))+geom_boxplot()
# Change Color
ggplot(dataset12, aes(Weight))+geom_boxplot(colour="red")
# Histogram
ggplot(dataset12, aes(Weight))+geom_histogram(colour="red")
ggplot(dataset12, aes(Weight))+geom_histogram(color="cyan")+ggtitle("The Title")
# Increase binwidth
ggplot(dataset12, aes(Weight))+geom_histogram(binwidth = 1)
# On the basis of Categorical data 
ggplot(dataset12, aes(Weight, fill=Gender))+geom_histogram()
# How to give title
ggplot(dataset12, aes(Weight, fill=Gender))+geom_histogram() + ggtitle("Title")
# Give Axis 
ggplot(dataset12, aes(Weight, fill=Gender))+geom_histogram() + ggtitle("Title")+labs(x="X Axis", y="Y Axix")
# Theme
ggplot(dataset12, aes(Weight, fill=Gender))+geom_histogram() + theme_replace()




