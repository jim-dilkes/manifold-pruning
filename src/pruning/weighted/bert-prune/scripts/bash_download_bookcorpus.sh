data_dir = '/content/bert-prune/data'

# # Download bookcorpus to drive
! wget https://battle.shawwn.com/sdb/books1/books1.tar.gz -P $data_dir
! gzip -dc < $data_dir/books1.tar.gz -c > $data_dir/books1.tar
! rm $data_dir/books1.tar.gz

# Extract bookcorpus tar from drive to local (~2min)
book_dir = '/content/bert-prune/data/bookcorpus'
# ! rm -r $book_dir
! mkdir -p $book_dir
! tar -xf $data_dir/books1.tar -C $book_dir
# ! rm -r $book_dir/out_txts
! mv $book_dir/books1/epubtxt $book_dir/out_txts
! rm -r $book_dir/books1

# Dummy wiki data folder
wiki_dir = '/content/bert-prune/data/enwiki'
! mkdir -p $wiki_dir