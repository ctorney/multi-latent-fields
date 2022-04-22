

## Set Up
{
  library(readr)
  library(sp)
  library(rgeos)
  library(INLA)
  library(inlabru) 
  library(spatstat)
  library(ggplot2)
  library(viridis)
  library(scales)
  #init.tutorial()
}

filelist<- list.files('../data/simdata/',pattern="^sim",full.names=TRUE)
df <- as.data.frame(matrix(NA,nrow=length(filelist),ncol=4))

for (i in 1:length(filelist)){
  df[i,1]=filelist[i]

## Format Point Data
simdata <- read.csv(filelist[i])
simdata$x<-simdata$x/1000
simdata$y<-simdata$y/1000
simdata$photo_area<-simdata$photo_area/(1000*1000)



# we stored the true count in transect id -1
true_count <- subset(simdata,transect_id<0)$wildebeest

# retain all other 'real' transects
simdata <- subset(simdata,transect_id>0)

PointCoords <- matrix(c(simdata$x, simdata$y), ncol = 2, byrow = FALSE, dimnames = list(NULL, c("Easting", "Northing")))
SpPoints <- SpatialPoints(coords = PointCoords, bbox = NULL, proj4string = CRS("+proj=utm +zone=37S +datum=WGS84")) ## Just guessing at the projection here...
photo_spdf <- SpatialPointsDataFrame(SpPoints, simdata)
mesh.loc <- as(photo_spdf, "SpatialPoints")


boundary <- inla.nonconvex.hull(mesh.loc, convex = -0.01,resolution=c(143,143),concave=-0.2)
boundary_p <- SpatialPolygons(Srl = list(Polygons(srl = list(Polygon(coords = boundary$loc)), ID = "a")), proj4string = CRS("+proj=utm +zone=37S +datum=WGS84"))

mesh <- inla.mesh.2d(loc=mesh.loc, offset=-0.05,max.edge = c(5, 5),cutoff = 0.45)



## Define SPDE
matern <- inla.spde2.pcmatern(mesh, alpha = 2, prior.range = c(0.5, 0.0001), prior.sigma = c(2,0.001))

## Specify model components 
cmp <- wildebeest ~ field(map = coordinates, model = matern) + Intercept

## Fit model
fit <- bru(cmp, family = "Poisson", data = photo_spdf, options=list(E=simdata$photo_area))
summary(fit)


abundance <- predict(fit,ipoints(boundary_p, mesh),~ sum(weight * exp(field+ Intercept)))
df[i,2]=abundance$mean
df[i,3]=abundance$q0.025
df[i,4]=abundance$q0.975
write.csv(df,file="bru_output.txt")

}
  


