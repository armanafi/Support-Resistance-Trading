import pandas as pd
import numpy as np 
import plotly.graph_objects as go
from binance import Client as Client_binance
import sys
import time
import os

""" 
Description: Find and Plot Resistance/Support levels
Disclaimer: The following code is intended for educational purposes only and shall not to be considered as trading tools
Author = Ar. Manafi
Version = 0.1
Copyright (c) 2024 Uniarma GmbH, all rights reserved
"""

class FindLevels():
  """ Find levels of Resistance/Support """
  __slots__ = ("back_data", "radius", "levels", "levels_shape", "level_window_high", "level_window_low", "level_window_max", "level_window_min", "level_window_max_counter","level_window_min_counter","level_trend","level_trend_num")
  
  def __init__(self, radius=240):  
    self.radius=radius
    #---------------configuration, must be updated every midnight, with last day prices
    self.levels_shape = np.empty(shape=[0, 4]) #CurrentTime, FindTime, Level
    self.levels = pd.DataFrame({'level_max':[],'level_min':[],'occurrences':[],'first_time':[],'last_time':[]})
    self.levels=self.levels.astype({'level_max':'float','level_min':'float','occurrences':'int','first_time':'str','last_time':'str'})

    self.level_window_high=[0]*radius
    self.level_window_low=[0]*radius
    self.level_window_max=0
    self.level_window_min=10**10#a big number
    self.level_window_max_counter=0
    self.level_window_min_counter=0
    self.level_trend=0
    self.level_trend_num=0

  def find_fast_new_levels(self, cur_data, back_data, i): 
    """ fast finding new levels S/R by looking for highest/lowest prices in both sides in distance=radius """
    level=0
    positive_signal=False
    cur_close=cur_data.Close

    #find the max/min in first Radius
    cur_high=cur_data.High  
    if self.level_window_max<cur_high:
      self.level_window_max=cur_high
      self.level_window_max_counter=0

    cur_low=cur_data.Low  
    if cur_low<self.level_window_min:
      self.level_window_min=cur_low
      self.level_window_min_counter=0

    #find next resistance/support level
    if i>2*self.radius:#at the start of the program, we must have enough data
      if self.level_window_high[self.radius-1]==self.level_window_max and self.level_window_max_counter==self.radius:#detect rsistance level at the end of second radius
          level=self.level_window_max
          self.level_window_max=0
      if self.level_window_low[self.radius-1]==self.level_window_min and self.level_window_min_counter==self.radius:#detect support level at the end of second radius
          level=self.level_window_min
          self.level_window_min=10**10#a big number

    #shift right and insert new prices
    self.level_window_high=self.level_window_high[:-1] #shift right
    self.level_window_high.insert(0,cur_high)  
    self.level_window_low=self.level_window_low[:-1] #shift right
    self.level_window_low.insert(0,cur_low)  

    self.level_window_max_counter+=1
    self.level_window_min_counter+=1

    #update levels history
    if level:#if found new level
      
      if self.levels_shape.size>0:
        plevel=self.levels_shape[-1,2] #previous level
        positive_signal=level*1.003<plevel<cur_close/1.003<plevel*1.01 #can be used for trading

      # Update data for displaying Support/Resiatance levels
      self.levels_shape = np.append(self.levels_shape, [[cur_data.Time, back_data.Time, level, positive_signal]], axis=0)

      # Find new level position in Dataframe levels
      i_level = np.searchsorted(self.levels['level_max'].to_numpy(dtype='float'), level)      
      level_distance_min= cur_close*fee_rate
      num_levels=len(self.levels)
      
      #look back and check if near to previuos level
      if i_level!=0 and level-self.levels.level_min.iloc[i_level-1]<level_distance_min\
      and i_level!=num_levels and level-self.levels.level_min.iloc[i_level-1]<self.levels.level_max.iloc[i_level]-level:#more near to lower channel than upper channel, then merge in it
        if self.levels.level_max.iloc[i_level-1]<level:
          self.levels.at[i_level-1,'level_max']=level #change the previous level max

        self.levels.at[i_level-1,'occurrences']+=1 
        self.levels.at[i_level-1,'last_time']=back_data.Time

      elif i_level!=num_levels and self.levels.level_max.iloc[i_level]-level<level_distance_min: #look forward and check if near to next level, then merge in it
        if level<self.levels.level_min.iloc[i_level]:
          self.levels.at[i_level,'level_min']=level #change the next level min

        self.levels.at[i_level,'occurrences']+=1 
        self.levels.at[i_level,'last_time']=back_data.Time

      else:#new channel of levels
        level_df=pd.DataFrame({'level_max':[level], 'level_min':[level], 'occurrences':1, \
                              'first_time':[back_data.Time], 'last_time':[back_data.Time]  })
        if num_levels:#if levels is not empty
          self.levels=pd.concat([self.levels.loc[:i_level-1], level_df, self.levels.loc[i_level:]], ignore_index=True) #insert new level and reset indexs
        else:
          self.levels=level_df

    return positive_signal #levels


class BackData():
  """ loads/downloads required back data """
  __slots__ = ("back_data", "back_file_name", "download_back_days") #use slots to reduce RAM overhead
  def __init__(self, back_file_name="ETHUSDT.txt", download_back_days=30):
      self.back_data=pd.DataFrame()
      self.back_file_name = back_file_name
      self.download_back_days = download_back_days
      self.get_back_data()
  
  def get_back_data(self):
    """ load back data from file"""
    #if back data file is not exist then fetch latest days back data from the Binance
    if not os.path.isfile(self.back_file_name): 
      self.download_last_data_binace()
    self.back_data = pd.read_csv(self.gap_fill(), index_col=0) #read back data file
    self.back_data['Time']= pd.to_datetime(self.back_data.Time, unit='ms')

  def download_last_data_binace(self):
    """ back data load or fetch from Binance """
    client = Client_binance('','')
    self.back_data = pd.DataFrame(client.get_historical_klines(symbol, '1m', str(self.download_back_days)+' day ago UTC')) #Get Historical Klines from Binance(last 100 min bars), and put in a Panda Matrix (Time'ms', OHLCV, ..)
    self.back_data = self.back_data.iloc[:,:6] # trim first 6 columns(time+OHLCV )
    self.back_data.columns = ['Time','Open', 'High', 'Low', 'Close', 'Volume'] #define Panad column names
    self.back_data.to_csv(self.back_file_name) #save to file for next Runs  

  def gap_fill(self): 
      """ fill any possible gap in back data """
      gap_fillded_file_name=self.back_file_name[:-4]+'-Gap Filled.txt'
      fileIn = open(self.back_file_name, 'r')
      lines = fileIn.readlines()
      fileIn.close()

      time_index=lines[1].find(',')+1
      current_time=int(lines[1][time_index:time_index+13]) #time of first line 

      linesNum=len(lines)

      if lines[0][:4]!='G.F.':
          out_lines='G.F.'+lines[0]+lines[1]

          for count in range (1, linesNum):
              time_index=lines[count].find(',')+1
              next_time=int(lines[count][time_index:time_index+13]) #time of next line 
              gap=int(next_time/60000)-int(current_time/60000) 
              
              if gap<0:
                  print("Error Gap Fill: repetetive data from previous seconds!")
                  sys.exit()            
              
              for i in range(0, gap):
                  out_lines+= lines[count]

              current_time=next_time
              count+=gap 
              
              if count%100000==0:
                  print(count)

          fileOut = open(gap_fillded_file_name, 'w')
          lines = fileOut.writelines(out_lines)
          fileOut.flush() 
          fileOut.close()
          return gap_fillded_file_name
      else:
          return self.back_file_name

class Plot():
  """ plot Close/Time + Resitance/Support levels"""
  def __init__(self, time, close , levels, title):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=list(time), y=list(close), line=dict(color='Blue', width=1 )))#Display Close Prices

    for i in range(1,len(levels)):# Display Resistance/Support Leels
      if levels[i,3]:
        color='Green'
      else:
        color="Red"  
      figure.add_shape(type='line', x0=levels[i,1], y0=levels[i,2], x1=levels[i,0], y1=levels[i,2], line=dict(color=color, width=1 ))

    # Set the title text of figure
    figure.update_layout(title_text="Detected Support/Resitance Levels", xaxis_title="Time", yaxis_title=title)

    # Add range slider
    figure.update_layout(
        xaxis=dict(rangeselector=dict(
                buttons=list([dict(count=1, label="1m", step="month", stepmode="backward"),
                              dict(count=6, label="6m", step="month", stepmode="backward"),
                              dict(count=1, label="YTD",step="year",  stepmode="todate"),
                              dict(count=1, label="1y", step="year",  stepmode="backward"),
                              dict(step="all") ])),
                rangeslider=dict(visible=True), type="date"))

    figure.update_layout(yaxis=dict(autorange=True, fixedrange=False))
    figure.show()

#************************************************ main body ***********************************************
symbol='ETHUSDT'
download_back_days=30 # maximum 30 days, download from Binance. recommended if no backdata file is available
radius=240 # must be higher/lower than how many points on the left and right side

back_data=BackData(back_file_name=symbol+'.txt' ,download_back_days=download_back_days).back_data 

fL=FindLevels(radius) #for finding levels support/resistance

trades = pd.DataFrame({'buy_price':[],'sell_price':[],'pattern':[],'buy_stime':[],'sell_stime':[],'pL0':[],'pL1':[],'pL2':[],'pL3':[],'pL4':[],'pL5':[],'pL6':[],'pL7':[],'pL8':[],'pL9':[],'level_trend':[],'level_trend_num':[],'roi_max':[],'roi_min':[],'roi':[]})
trades=trades.astype({'buy_price':'float','sell_price':'float','pattern':'str','buy_stime':'str','sell_stime':'str','pL0':'float','pL1':'float','pL2':'float','pL3':'float','pL4':'float','pL5':'float','pL6':'float','pL7':'float','pL8':'float','pL9':'float','level_trend':'float','level_trend_num':'float','roi_max':'float','roi_min':'float','roi':'float'})

fee_rate=0.0015 #Binance fees
profit_rate_min=0.005 # for coveing 0.15% Binance fees + 0.35% minimum net profit   

bought_status=''
cur_price=0
inv_profit=1 #our first investment + its profit/losses in the trades
i_trade=0 #last trade index in 'trades' data frame

roi=0
roi_max=0
roi_min=0

start_time=time.perf_counter()
len_frame=back_data.shape[0]
for i in range(len_frame): #look back with shift back radius  
  
  positive_signal=fL.find_fast_new_levels(back_data.iloc[i], back_data.iloc[i-radius-1], i) 
    
  #-------------- put trading code here, the follwing code is only an educational sample---------------
  pcur_price=cur_price
  cur_price=back_data.Close.iloc[i]
  cur_time=back_data.Time.iloc[i]
  num_levels=len(fL.levels_shape)
  if num_levels>0:#if there is at least 2 levels
    if bought_status=='' and positive_signal:       
      level=fL.levels_shape[-1,2]
      plevel=fL.levels_shape[-2,2]

      buy_price=cur_price
      sell_price=cur_price+cur_price-plevel#-fL.levels_shape[-1,2]
      take_profit=cur_price*1.10
      stop_loss=plevel#max(cur_price*.97, fL.levels_shape[-2,2])#max(.93*cur_price, level_min)
    
      trades.at[i_trade,'buy_price']=cur_price
      trades.at[i_trade,'buy_stime']=cur_time
      for i in range(0, min(num_levels,10)):#record 10 recent levels if exist
        trades.at[i_trade,'pL'+str(i)]=fL.levels_shape[-i-1,2]
      trades.at[i_trade,'level_trend']=fL.level_trend
      trades.at[i_trade,'level_trend_num']=fL.level_trend_num
      bought_status='Buy'
      
      fL.level_trend=0
      fL.level_trend_num=0

    elif bought_status=='Buy':
      roi=(cur_price/buy_price-1-fee_rate)*100 
      if roi_max<roi:
        roi_max=roi
      if roi_min>roi:
        roi_min=roi

      trades.at[i_trade,'pattern']=''
      if cur_price<sell_price<=pcur_price:#cross down 
       trades.at[i_trade,'pattern']='L1'
      if pcur_price<take_profit<=cur_price:# or cur_price<np.mean(back_data.Close.iloc[i-5:i+1].to_numpy()):#cross up from TP
       trades.at[i_trade, 'pattern']='TP'         
      if cur_price<stop_loss<=pcur_price :#cross down from SL
       trades.at[i_trade, 'pattern']='SL'
      if roi_max<0 and roi<-1:
       trades.at[i_trade, 'pattern']='R-'

      if trades.pattern.loc[i_trade]: # matched for sell condition?
       trades.at[i_trade, 'sell_price']=cur_price
       trades.at[i_trade, 'sell_stime']=cur_time       
       trades.at[i_trade, 'roi_max']=round(roi_max,2)
       trades.at[i_trade, 'roi_min']=round(roi_min,2)
       trades.at[i_trade, 'roi']=round(roi,2)
       inv_profit*=(1+roi/100)
       print('(', trades.buy_stime.iloc[i_trade], '=>', cur_time,')', plevel, level, buy_price, cur_price, round(roi_min,2), round(roi_max,2), round(roi,2), round((inv_profit-1)*100,2))
       i_trade+=1
       bought_status=''
       roi=0
       roi_min=0
       roi_max=0
print('Run time find levels+trades:',time.perf_counter()-start_time,'seconds', 'profit:', round(inv_profit-1,3))  


trades.to_csv('trades.csv')#save trades

fL.levels.to_csv('levels.csv')#save levels

plot= Plot(back_data.Time, back_data.Close, fL.levels_shape, symbol) #plot Close prices + found levels

print('terminated')