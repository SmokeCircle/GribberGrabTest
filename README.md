# GribberGrabTest
忽略追踪以下目录，可以在 ignore 文件下修改：

# ignore files 
/data/* 
/dump/* 
/output/* 
/publish/*  
/weights/*  下三个文件best-model_epoch-140_mae-0.0020_loss-0.0082.pth  gribber_kernels_sort.pth gribber_kernels_stack.pth 超过100M 也没有归入跟踪管理，更新时需手动替换

用于本地测试时注意修改：DBHelper 的数据库地址 
用于本地测试时注意修改api2: 
os.chdir("/data/GribberGrabTest") 
 rootDir =  "/data/GribberGrabTest/data/dxf" 

用于本地测试时注意修改api3:  
os.chdir("/data/GribberGrabTest") 

