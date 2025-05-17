sources = [
    "RedPajamaCommonCrawl",#train on some of this first 10%
    "RedPajamaC4",#7th
    "RedPajamaStackExchange",#4th
    "RedPajamaWikipedia",#3rd
    "RedPajamaGithub",#6th
    "RedPajamaArXiv",#5th
    "RedPajamaBook" #2nd
]

filtered_datasets = {}

for source in sources:
    print(f"Filtering: {source}")
    filtered = data.filter(lambda x: x['meta']['redpajama_set_name'] == source)
    filtered_datasets[source] = filtered
    filtered.save_to_disk(f"./slimpajama_filtered/{source}")