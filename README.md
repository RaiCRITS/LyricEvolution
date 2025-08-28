## Songs JSON Format

The dataset should be structured as a dictionary where **each key is a year** (as a string), and the value is a list of song entries for that year. Each song entry is a dictionary. The **only required fields** are:

| Key | Type | Description |
|-----|------|-------------|
| `song` | string | Full song name including performer(s) (e.g., `"Achille Togliani - Mai più"`). |
| `text` | string | Full lyrics of the song. |

Optional fields:

| Key | Type | Description |
|-----|------|-------------|
| `title` | string | Song title only (e.g., `"Mai più"`). |
| `authors` | string | Comma- or dash-separated names of the authors/composers. |
| `performers` | string | Names of the performers. |
| `text_link` | string | URL to the source of the lyrics. |
| `text_segments` | list of strings | Segments or lines of the song lyrics (split for easier processing). If not provided, the framework will automatically generate these segments from the full text. |

### Example

```json
{
  "1951": [
    {
      "song": "Achille Togliani - Mai più",
      "title": "Mai più",
      "authors": "Fuselli - Rolando",
      "performers": "Achille Togliani",
      "text": "Felicitá, dolce amor, primavera...\nOra è finita...",
      "text_link": "https://www.leparoledisanremo.it/canzoni/mai-piu/",
      "text_segments": [
        "Felicitá, dolce amor, primavera...",
        "Ora è finita: cosa resta in me?...",
        "Quindi mai, mai più, mai più d'amare tenterò..."
      ]
    },
    {
      "song": "Achille Togliani - Mani che si cercano",
      "title": "Mani che si cercano",
      "authors": "Redi - Colombi",
      "performers": "Achille Togliani",
      "text": "Vorrei la vita fermar, con te per sempre restar...",
      "text_link": "https://www.leparoledisanremo.it/canzoni/mani-che-si-cercano/"
      // text_segments can be omitted; the framework will generate them automatically
    }
  ]
}
