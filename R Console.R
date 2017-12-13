
R version 3.4.2 (2017-09-28) -- "Short Summer"
Copyright (C) 2017 The R Foundation for Statistical Computing
Platform: x86_64-apple-darwin15.6.0 (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

During startup - Warning messages:
1: Setting LC_CTYPE failed, using "C" 
2: Setting LC_COLLATE failed, using "C" 
3: Setting LC_TIME failed, using "C" 
4: Setting LC_MESSAGES failed, using "C" 
5: Setting LC_MONETARY failed, using "C" 
[R.app GUI 1.70 (7434) x86_64-apple-darwin15.6.0]

WARNING: You're using a non-UTF8 locale, therefore only ASCII characters will work.
Please read R for Mac OS X FAQ (see Help) section 9 and adjust your system preferences accordingly.
> setwd()
Error in setwd() : argument "dir" is missing, with no default
> getwd()
[1] "/Users/Vinit"
> setwd('/Users/Vinit/Documents/PROJECTS_MY/IMDB_MYPROJECT/extras')
> require(stringr)
Loading required package: stringr
> WordList <- str_split(readLines("output.txt"), pattern = " ")
Warning message:
In readLines("output.txt") : incomplete final line found on 'output.txt'
> text<-paste(unlist(WordList), collapse=' ')
> str(text)
 chr "[\"This movie takes the Golem from the 20s movies and puts him in a museum where he is found by Roddy McDowall'"| __truncated__
> 
> library(dplyr)

Attaching package: 'dplyr'

The following objects are masked from 'package:stats':

    filter, lag

The following objects are masked from 'package:base':

    intersect, setdiff, setequal, union

> text_df <- data_frame(line = 1, text = text)
> text_df
# A tibble: 1 x 2
   line
  <dbl>
1     1
# ... with 1 more variables: text <chr>
> 
> library(tidytext)
> 
> text_df2 <- text_df %>%
+   unnest_tokens(word,text)
> 
> data(stop_words)
> 
> text_df2 <- text_df2 %>%
+   anti_join(stop_words)
Joining, by = "word"
> 
> tibble<-text_df2 %>%
+   count(word,sort=TRUE)
> 
> tibblefiltered = tibble %>% filter(n > 4)
> 
> #Wordclouds
> library(wordcloud)
Loading required package: RColorBrewer
> text_df2 %>%
+   anti_join(stop_words) %>%
+   count(word) %>%
+   with(wordcloud(word, n, max.words = 100, colors=brewer.pal(8, "Dark2")))
Joining, by = "word"
> 
> 
> barplot(tibblefiltered[1:2]$n, las = 2, names.arg = tibblefiltered[1:2]$word,
+         col ="maroon", main ="Most frequent words",
+         ylab = "Word frequencies")
> library(reshape2)
> text_df2 %>%
+   filter()
# A tibble: 599 x 2
    line       word
   <dbl>      <chr>
 1     1      movie
 2     1      takes
 3     1      golem
 4     1        20s
 5     1     movies
 6     1     museum
 7     1      found
 8     1      roddy
 9     1 mcdowall's
10     1  character
# ... with 589 more rows
> text_df2 %>%
+   inner_join(get_sentiments("bing")) %>%
+   count(word, sentiment, sort=TRUE) %>%
+   acast(word ~ sentiment, value.var = "n", fill = 0) %>%
+   comparison.cloud(colors = c("darkred", "darkgreen"), max.words=100)
Joining, by = "word"
> library(ggplot2)
> library(janeaustenr)
> library(tidyr)

Attaching package: 'tidyr'

The following object is masked from 'package:reshape2':

    smiths

> bing_word_counts <- text_df2 %>%
+   inner_join(get_sentiments("bing")) %>%
+   count(word, sentiment, sort = TRUE) %>%
+   ungroup()
Joining, by = "word"
> bing_word_counts %>%
+   group_by(sentiment) %>%
+   top_n(10) %>%
+   ungroup() %>%
+   mutate(word = reorder(word, n)) %>%
+   ggplot(aes(word, n, fill = sentiment)) +
+   geom_col(show.legend = FALSE) +
+   facet_wrap(~sentiment, scales = "free_y") +
+   labs(y = "Contribution to sentiment",
+        x = NULL) +
+   coord_flip()
Selecting by n
> 
> 
> 