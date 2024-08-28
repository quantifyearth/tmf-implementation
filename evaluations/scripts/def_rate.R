


def_rate <- function(data,t0,period_length,process='all'){
  
  # get name of column for start year
  
  t0_index <- grep(paste0('luc_',t0),colnames(data))
  
  # filter down to pixels with undisturbed forest (JRC class 1)
  
  data_filtered <- data[data[,t0_index]==1,] 
  
  # count 1s at t0 in project and match
  
  proj_1s <- data_filtered %>% filter(type=='Project') %>% nrow() 
  cf_1s <- data_filtered %>% filter(type=='Counterfactual') %>% nrow() 
  
  # identify where there have been changes during the evaluation period
  
  tend <- t0 + period_length
  
  luc_tend <- data_filtered %>% 
    select(paste0('luc_',tend))
  
  # choosing processes to measure
  
  if(process=='def_only'){
    
    response <- case_when(
      luc_tend==1 ~ 0,
      luc_tend==2 ~ 0,
      luc_tend==3 ~ 1,
      luc_tend==4 ~ 0,
      luc_tend>4 ~ 0)
    
  } else if(process=='deg_only'){
      
      response <- case_when(
        luc_tend==1 ~ 0,
        luc_tend==2 ~ 1,
        luc_tend==3 ~ 0,
        luc_tend==4 ~ 0,
        luc_tend>4 ~ 0)
      
    } else {
    
    response <- case_when(
      luc_tend==1 ~ 0,
      luc_tend==2 ~ 1,
      luc_tend==3 ~ 1,
      luc_tend==4 ~ 1,
      luc_tend>4 ~ 0)
    
    }
  
  
  data_filtered$response <- response
  
  # count up number of pixels where there have been changes for each type
  
  proj_changes <- data_filtered %>% filter(response==1 & type=='Project') %>% 
    nrow()
  cf_changes <- data_filtered %>% filter(response==1 & type=='Counterfactual') %>% 
    nrow()
  
  # calculate deforestation rate (= the rate of loss of undisturbed forest) as a percentage
  
  proj_rate <- 100*(proj_changes/proj_1s)/period_length
  cf_rate <- 100*(cf_changes/cf_1s)/period_length
  
  # make df
  
  df <- data.frame(matrix(ncol=2,nrow=1))
  colnames(df) <- c('Project','Counterfactual')
  df[1,1] <- proj_rate
  df[1,2] <- cf_rate
  
  return(df)
  
}



def_rate_seperate <- function(data,t0,period_length){
  
  # get name of column for start year
  
  t0_index <- grep(paste0('luc_',t0),colnames(data))
  
  # filter down to pixels with undisturbed forest (JRC class 1)
  
  data_filtered <- data[data[,t0_index]==1,] 
  
  # count 1s at t0 in project and cf
  
  proj_1s <- data_filtered %>% filter(type=='Project') %>% nrow() 
  cf_1s <- data_filtered %>% filter(type=='Counterfactual') %>% nrow() 
  
  # identify where there have been changes during the evaluation period
  
  tend <- t0 + period_length
  
  luc_tend <- data_filtered %>% 
    select(paste0('luc_',tend))
  
  # measuring responses
  
  def_response <- case_when(
    luc_tend==1 ~ 0,
    luc_tend==2 ~ 0,
    luc_tend==3 ~ 1,
    luc_tend==4 ~ 0,
    luc_tend>4 ~ 0)
  
  deg_response <- case_when(
    luc_tend==1 ~ 0,
    luc_tend==2 ~ 1,
    luc_tend==3 ~ 0,
    luc_tend==4 ~ 0,
    luc_tend>4 ~ 0)
  
  ref_response <- case_when(
    luc_tend==1 ~ 0,
    luc_tend==2 ~ 0,
    luc_tend==3 ~ 0,
    luc_tend==4 ~ 1,
    luc_tend>4 ~ 0)
  
  data_filtered$def_response <- def_response
  data_filtered$deg_response <- deg_response
  data_filtered$ref_response <- ref_response
  
  # count up number of pixels where there have been changes for each type
  
  proj_def_changes <- data_filtered %>% filter(def_response==1 & type=='Project') %>% 
    nrow()
  cf_def_changes <- data_filtered %>% filter(def_response==1 & type=='Counterfactual') %>% 
    nrow()
  
  proj_deg_changes <- data_filtered %>% filter(deg_response==1 & type=='Project') %>% 
    nrow()
  cf_deg_changes <- data_filtered %>% filter(deg_response==1 & type=='Counterfactual') %>% 
    nrow()
  
  proj_ref_changes <- data_filtered %>% filter(ref_response==1 & type=='Project') %>% 
    nrow()
  cf_ref_changes <- data_filtered %>% filter(ref_response==1 & type=='Counterfactual') %>% 
    nrow()
  
  # calculate deforestation rate (= the rate of loss of undisturbed forest) as a percentage
  
  proj_def <- 100*(proj_def_changes/proj_1s)/period_length
  cf_def <- 100*(cf_def_changes/cf_1s)/period_length
  
  proj_deg <- 100*(proj_deg_changes/proj_1s)/period_length
  cf_deg <- 100*(cf_deg_changes/cf_1s)/period_length
  
  proj_ref <- 100*(proj_ref_changes/proj_1s)/period_length
  cf_ref <- 100*(cf_ref_changes/cf_1s)/period_length
  
  # adding the degraded-to-deforested transition
  
  data_filtered_2 <- data[data[,t0_index]==2,]
  
  # count 2s at t0 in project and cf
  
  proj_2s <- data_filtered_2 %>% filter(type=='Project') %>% nrow()
  cf_2s <- data_filtered_2 %>% filter(type=='Counterfactual') %>% nrow() 
  
  # identify where there have been changes during the evaluation period
  
  luc_tend_2 <- data_filtered_2 %>% 
    select(paste0('luc_',tend))
  
  def_response_2 <- case_when(
    luc_tend_2==1 ~ 0,
    luc_tend_2==2 ~ 0,
    luc_tend_2==3 ~ 1,
    luc_tend_2==4 ~ 0,
    luc_tend_2>4 ~ 0)
  
  data_filtered_2$def_response_2 <- def_response_2
  
  proj_def_changes_2 <- data_filtered_2 %>% filter(def_response_2==1 & type=='Project') %>% 
    nrow()
  cf_def_changes_2 <- data_filtered_2 %>% filter(def_response_2==1 & type=='Counterfactual') %>% 
    nrow()
  
  proj_deg_to_def <- 100*(proj_def_changes_2/proj_2s)/period_length
  cf_deg_to_def <- 100*(cf_def_changes_2/cf_2s)/period_length
  
  # make df
  
  df <- data.frame(matrix(ncol=4,nrow=8))
  
  colnames(df) <- c('Process','Forest type','Location','Rate (%/year)')
  
  df[1] <- c(rep(c('Degradation','Deforestation','Deforestation','Reforestation'),each=2))
  df[2] <- c(rep(c('Undisturbed forest','Undisturbed forest','Disturbed forest','Undisturbed forest'),each=2))
  df[3] <- c(rep(c('Project','Counterfactual'),times=4))
  df[4] <- c(proj_deg,cf_deg,proj_def,cf_def,proj_deg_to_def,cf_deg_to_def,proj_ref,cf_ref)
  
  return(df)
  
}

get_prop_class <- function(data,t0,class){
  
  t0_index <- grep(paste0('luc_',t0),colnames(data))
  data_filtered <- data[data[,t0_index]==class,]
  
  total_count <- data %>% nrow()
  class_count <- data_filtered %>% nrow()
  prop <- class_count/total_count
  
  return(prop)
  
}


def_rate_single <- function(data,t0,period_length){
  
  # get name of column for start year
  
  t0_index <- grep(paste0('luc_',t0),colnames(data))
  
  # filter down to pixels with undisturbed forest (JRC class 1)
  
  data_filtered <- data[data[,t0_index]==1,] 
  
  # count 1s at t0 in project and cf
  
  no_1s <- nrow(data_filtered)
  
  # identify where there have been changes during the evaluation period
  
  tend <- t0 + period_length
  
  luc_tend <- data_filtered %>% 
    select(paste0('luc_',tend))
  
  # measuring responses
  
  def_response <- case_when(
    luc_tend==1 ~ 0,
    luc_tend==2 ~ 0,
    luc_tend==3 ~ 1,
    luc_tend==4 ~ 0,
    luc_tend>4 ~ 0)
  
  deg_response <- case_when(
    luc_tend==1 ~ 0,
    luc_tend==2 ~ 1,
    luc_tend==3 ~ 0,
    luc_tend==4 ~ 0,
    luc_tend>4 ~ 0)
  
  ref_response <- case_when(
    luc_tend==1 ~ 0,
    luc_tend==2 ~ 0,
    luc_tend==3 ~ 0,
    luc_tend==4 ~ 1,
    luc_tend>4 ~ 0)
  
  data_filtered$def_response <- def_response
  data_filtered$deg_response <- deg_response
  data_filtered$ref_response <- ref_response
  
  # count up number of pixels where there have been changes for each type
  
  def_changes <- data_filtered %>% filter(def_response==1) %>% 
    nrow()
  
  deg_changes <- data_filtered %>% filter(deg_response==1) %>% 
    nrow()
  
  ref_changes <- data_filtered %>% filter(ref_response==1) %>% 
    nrow()
  
  # calculate deforestation rate (= the rate of loss of undisturbed forest) as a percentage
  
  def <- 100*(def_changes/no_1s)/period_length
  
  deg <- 100*(deg_changes/no_1s)/period_length
  
  ref <- 100*(ref_changes/no_1s)/period_length
  
  # adding the degraded-to-deforested transition
  
  data_filtered_2 <- data[data[,t0_index]==2,]
  
  # count 2s at t0 in project and cf
  
  no_2s <- data_filtered_2 %>% nrow()
  
  # identify where there have been changes during the evaluation period
  
  luc_tend_2 <- data_filtered_2 %>% 
    select(paste0('luc_',tend))
  
  def_response_2 <- case_when(
    luc_tend_2==1 ~ 0,
    luc_tend_2==2 ~ 0,
    luc_tend_2==3 ~ 1,
    luc_tend_2==4 ~ 0,
    luc_tend_2>4 ~ 0)
  
  data_filtered_2$def_response_2 <- def_response_2
  
  def_changes_2 <- data_filtered_2 %>% filter(def_response_2==1) %>% 
    nrow()

  deg_to_def <- 100*(def_changes_2/no_2s)/period_length
  
  # make df
  
  df <- data.frame(matrix(ncol=3,nrow=4))
  
  colnames(df) <- c('Process','Forest type','Rate (%/year)')
  
  df[1] <- c('Degradation','Deforestation','Deforestation','Reforestation')
  df[2] <- c('Undisturbed forest','Undisturbed forest','Disturbed forest','Undisturbed forest')
  df[3] <- c(deg,def,deg_to_def,ref)
  
  return(df)
  
}