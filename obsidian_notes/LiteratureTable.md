You can change the source code of the table to customize the table

```dataview
table title, authors, year, join(map(tags, (t) => " <span class='tag-style'>#" + t + "</span> "), " ") as tags
from "03_LiteratureCollection"
sort year desc
```



