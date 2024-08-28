
std_mean_diff <- function(path_to_pairs) {
  
  # clean data

  files_full_raw <- list.files(path_to_pairs,
                               pattern='*.parquet',full.names=T,recursive=F)
  files_full <- files_full_raw[!grepl('matchless',files_full_raw)]
  files_short_raw <- list.files(path=path_to_pairs,
                                pattern='*.parquet',full.names=F,recursive=F)
  files_short <- files_short_raw[!grepl('matchless',files_short_raw)]
  
  # initialise dfs
  
  vars <- c(colnames(read_parquet(files_full[1])),'pair')
  df <- data.frame(matrix(ncol=length(vars),nrow=0))
  colnames(df) <- vars
  
  for(j in 1:length(files_full)){
    
    # read in all parquet files for a given project
    
    f <- data.frame(read_parquet(files_full[j]))
    
    # append data to bottom of df
    
    df <- bind_rows(df,f)
    
  }

  # calculate smd

  smd_results <- data.frame(variable = character(), smd = numeric(), stringsAsFactors = FALSE)
  
  variables <- c('cpc10_d','cpc5_d','cpc0_d',
                 'cpc10_u','cpc5_u','cpc0_u',
                 'access','slope','elevation')
    
    for (var in variables) {
      k_var <- df[[paste0("k_", var)]]
      s_var <- df[[paste0("s_", var)]]
      
      k_mean <- mean(k_var, na.rm = TRUE)
      s_mean <- mean(s_var, na.rm = TRUE)
      k_sd <- sd(k_var, na.rm = TRUE)
      s_sd <- sd(s_var, na.rm = TRUE)
      
      pooled_sd <- sqrt((k_sd^2 + s_sd^2) / 2)
      smd <- (k_mean - s_mean) / pooled_sd
      
      smd_results <- rbind(smd_results, data.frame(variable = var, smd = smd, stringsAsFactors = FALSE))
    }
    
  return(smd_results)
}
  
  