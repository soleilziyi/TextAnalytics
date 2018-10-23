library(rugarch)
library(data.table)

get_garch_vol=function(data){
  
  model=ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder =  c(1, 1),external.regressors),
    mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
    distribution.model = "norm"
  )
  
  modelfit=ugarchfit(spec=model,data=data,out.sample=0)
  
  modelfor=ugarchforecast(modelfit, data = NULL, n.ahead = 1, out.sample = 0)
  
  return(c(sigma(modelfor),coef(modelfit)[1]))
}

returns=returns[,rev(names(returns)),with=FALSE]
returns=returns[,c(1:100),with=FALSE]

garch_vol=c()
garch_mean_return=c()
for(i in c(1:dim(returns)[1])){
  print(i)
  vol=try(as.numeric(get_garch_vol(as.numeric(unlist(returns[c(i),]))))[1])
  mu=try(as.numeric(get_garch_vol(as.numeric(unlist(returns[c(i),]))))[2])
  garch_vol[i]=vol
  garch_mean_return[i]=mu
}

garch_results=data.table("Vol"=garch_vol,
                         "Mean_Return"=garch_mean_return)
  
print(garch_vol)
write.csv(garch_results,file=paste(path,"/Garch_results.csv",sep=""),row.names=FALSE)

plot(modelfor)


path="C:/Users/ziyi/Desktop/Inde project/Independent Study Anseri-20170120T004329Z/Independent Study Anseri/8K data"
returns_raw=data.table(read.csv(file=paste(path,"/Price_ts_Info.csv",sep="")))
events_raw=data.table(read.csv(file=paste(path,"/events_ts_Info_new.csv",sep="")))

events_raw[,ITEM.1.01]

events=data.matrix(events_raw[,ITEM.1.01], rownames.force = NA)
returns=data.matrix(returns_raw[,Return], rownames.force = NA)

model=ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder =  c(1, 1),
                        external.regressors=events),
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
  distribution.model = "norm")

modelfit=ugarchfit(spec=model,data=returns,out.sample=0)

modelfor=ugarchforecast(modelfit, data = NULL, n.ahead = 1, out.sample = 0)

model_base=ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder =  c(1, 1)),
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
  distribution.model = "norm")

modelfit_base=ugarchfit(spec=model_base,data=returns,out.sample=0)
modelfor_base=ugarchforecast(modelfit, data = NULL, n.ahead = 1, out.sample = 0)

convergence(modelfit_base)
coef(modelfit_base)
infocriteria(modelfit_base)
sigma(modelfit_base)
fitted(modelfit_base)
residuals(modelfit_base)
uncvariance(modelfit_base)
uncmean(modelfit_base)
plot(modelfit_base)

convergence(modelfit)
coef(modelfit)
infocriteria(modelfit)
sigma(modelfit)
fitted(modelfit)
residuals(modelfit)
uncvariance(modelfit)
uncmean(modelfit)
plot(modelfit)

gd=ugarchdistribution(modelfit_base, n.sim = 500, recursive = TRUE, recursive.length = 6000, 
                   recursive.window = 500, m.sim = 100, solver = 'hybrid')
show(gd)
plot(gd, which = 1, window = 12)
plot(gd, which = 2, window = 12)
plot(gd, which = 3, window = 12)
plot(gd, which = 4, window = 12)

# c(sigma(modelfor),coef(modelfit)[1])
# 
# plot(modelfor)


