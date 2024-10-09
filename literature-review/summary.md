You can change the source code of the table to customize the table

```dataview
table title, authors, year, join(map(tags, (t) => " <span class='tag-style'>#" + t + "</span> "), " ") as tags
from "collection"
sort year desc
```



