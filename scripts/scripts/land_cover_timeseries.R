
get_luc_timeseries <- function(data,t0,tend,type='both'){
  
  years_list <- seq(t0,tend)

  if(type=='both'){

    df <- data.frame(matrix(ncol=4,nrow=8*length(years_list)))
  
    colnames(df) <- c('year','type','luc','percentage')
    
    counter <- 1
    
    for(year in years_list) {
      
      for(i in seq (1:4)) {
        
        for(type_value in c('Project','Counterfactual')) {
          
          total <- data %>% filter(type == type_value) %>% nrow()
          
          no_class_i <- data %>% filter(type == type_value & .data[[paste0('luc_',year)]]==i) %>% nrow()
          
          prop <- no_class_i/total
          
          df[counter,1] <- year
          df[counter,2] <- type_value
          df[counter,3] <- i
          df[counter,4] <- prop*100
          
          counter <- counter+1
        
        }
        
      }
      
    }

  } else if(type=='single'){

    df <- data.frame(matrix(ncol=3,nrow=4*length(years_list)))
  
    colnames(df) <- c('year','luc','percentage')
    
    counter <- 1
    
    for(year in years_list) {
      
      for(i in seq (1:4)) {
          
          total <- data %>% nrow()
          
          no_class_i <- data %>% filter(.data[[paste0('luc_',year)]]==i) %>% nrow()
          
          prop <- no_class_i/total
          
          df[counter,1] <- year
          df[counter,2] <- i
          df[counter,3] <- prop*100
          
          counter <- counter+1
        
        }
      
    }

  }

  return(drop_na(df))
  
}

luc_class1_uncertainty <- function(data,t0,tend) {

  years_list <- seq(t0-10,tend)
  
  df <- data.frame(matrix(ncol=4,nrow=2*length(unique(data$pair))*length(years_list)))
  
  colnames(df) <- c('year','type','pair','percent_class1')
  
  counter <- 1
  
  for(year in years_list) {
    
    for(type_value in c('Project','Counterfactual')) {
      
      for(pair_id in unique(data$pair)) {
        
        total <- data %>% filter(type == type_value & pair == pair_id) %>% nrow()
        
        no_class_i <- data %>% filter(type == type_value & pair == pair_id & .data[[paste0('luc_',year)]]==1) %>% nrow()
        
        prop <- no_class_i/total
        
        df[counter,1] <- year
        df[counter,2] <- type_value
        df[counter,3] <- pair_id
        df[counter,4] <- prop*100
        
        counter <- counter+1
        
      }
      
    }
    
  }
  
  return(drop_na(df))

}

