ls()
rm(list=ls())
search()


# loading packages
library(quantmod)
library(dplyr)
library(DBI)
library(odbc)
library(RMySQL)
library(mailR)
library(tableHTML)
library(formattable)

options(digits = 15)

#Conecting to Hedge

hedge <- dbConnect(MySQL(), user='root', password='gcemfwjszmilsIC9', dbname='public', host='35.223.74.197')

carteira<- dbGetQuery(hedge, 'SELECT * From carteira_fixa' )

cotas<-dbGetQuery(hedge, 'SELECT * From cotas' )

historico<-dbGetQuery(hedge, 'SELECT * From historico' )

caixa<-dbGetQuery(hedge, 'SELECT * From caixa' )

registro<-dbGetQuery(hedge, 'SELECT * From registro' )

dbDisconnect(hedge)

#Bringing Dollar and indexes last quote

ibov<-as.numeric(getQuote("^BVSP")[2])

dollar<-as.numeric(getQuote("BRL=X")[2])

sip<-as.numeric(getQuote("^GSPC")[2])

dow<-as.numeric(getQuote("^DJI")[2])

nasdaq<-as.numeric(getQuote("^IXIC")[2])


#Filtering BRL Stocks

carteira1<- carteira %>%
  filter(moeda =="BRL" & classe !="Caixa")

newprice<-c()

for (i in 1:nrow(carteira1)){
  
  id<-paste(carteira1$ticker[i],".SA",sep = "")
  
  teste<-as.numeric(getQuote(id)[2])
  
  newprice[i]<-teste
  
}

carteira1$value<-newprice

#Filtering US Stocks

carteira2<-carteira %>%
  filter(classe =="EI")

newprice<-c()

for (i in 1:nrow(carteira2)){
  
  id<-carteira2$ticker[i]
  
  teste<-as.numeric(getQuote(id)[2])
  
  newprice[i]<-teste
  
}

carteira2$value<-newprice


#Filtering Caixa

carteira3<-carteira %>%
  filter(classe =="Caixa") %>%
  inner_join(caixa)

carteira <- bind_rows(carteira1,carteira2,carteira3)

carteira$total<- ifelse(carteira$moeda=="BRL",carteira$value*carteira$quantidade,carteira$value*carteira$quantidade*dollar)

cota<-sum(carteira$total)/cotas$quantidade

cota1<-data.frame(cota)

cota1$ibov<-ibov
cota1$dollar<-dollar
cota1$dow<-dow
cota1$sip<-sip
cota1$nasdaq<-nasdaq

#Populating Historico

data<-Sys.Date() - 1
valor<-cota

valornorm<-(valor-historico$valor[1])/historico$valor[1]

ibovnorm<- (ibov-historico$ibov[1])/historico$ibov[1]

sipreturn<-(sip-historico$sip[1])/historico$sip[1]

dowreturn<-(dow-historico$dow[1])/historico$dow[1]

nasdaqreturn<-(nasdaq-historico$nasdaq[1])/historico$nasdaq[1]

historico2<-data.frame(cbind(valor,ibov,valornorm,ibovnorm,sip,sipreturn,dow,dowreturn,nasdaq,nasdaqreturn),stringsAsFactors = FALSE)

historico2$data<-data

#Print Current Results
round(100*((historico2[,c(1,2,5,7,9)]-historico[nrow(historico),c(1,2,5,7,9)])/historico[nrow(historico),c(1,2,5,7,9)]),2)

#result in cash
(cota-historico[nrow(historico),1])*registro$numcotas[1]

#Cashtotal

currency(cota*registro$numcotas)



