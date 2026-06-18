# CelebA Ground Truth Explorer

Dashboard locale per ispezionare `celeba_evaluation.json` usando la stessa indicizzazione dello split test di `torchvision.datasets.CelebA`.

## Avvio

Dalla cartella principale del progetto:

```bash
./ground_truth_dashboard/run_dashboard.sh
```

Poi apri:

```text
http://127.0.0.1:8876
```

Non richiede pacchetti Python aggiuntivi. Interrompi il server con `Ctrl+C`.

## Cosa mostra

- selezione delle 14 query del benchmark;
- navigazione avanti/indietro tra i source index ammessi dal JSON;
- ricerca diretta di un source index per la query selezionata;
- mappatura corretta tra dataset index e filename reale;
- immagine sorgente fissa e tutti i target validi, navigabili cinque alla volta;
- conteggio, evidenziazione e salto diretto ai target con lo stesso ID persona;
- attributi CelebA attivi, ID persona e verifica dei vincoli della query;
- distanza di Hamming sugli attributi non interrogati;
- matrice completa dei 40 attributi per il confronto.

I target mostrati sono ground truth validi, non risultati ordinati da un modello di retrieval.

## Percorsi alternativi

I percorsi predefiniti sono quelli del progetto. Possono essere sostituiti così:

```bash
./ground_truth_dashboard/run_dashboard.sh \
  --json /percorso/celeba_evaluation.json \
  --celeba-root /percorso/celeba \
  --port 9000
```
