# LLMs per tutti
Lo scopo di questo Readme è far capire informalmente a chiunque il funzionamento di base del progetto didattico, spiegando il codice che ho scritto nel main.py e i concetti alla base. Faremo prima una overview teorica, per poi passare a spiegare il codice del progetto.  

# Introduzione - Breve Overview sugli LLM
Cosa accade da quando scriviamo un prompt a quando riceviamo il nostro output del modello? Inizia tutto dalla tokenizzazione: il nostro prompt è una stringa che viene convertita in una sequenza di token numerici (ID interi). Più avanti spieghiamo meglio questo procedimento, ora diamo una introduzione generica.

Quindi, questa sequenza di token viene trasformata in una matrice di vettori, l'Embedding. Ogni token viene trasformato in un vettore di embedding, per cui la frase(il prompt) diventa una matrice composta da vettori di embedding.
Un embedding contiene numeri decimali (vettori). Ciascuno di questi numeri rappresenta una specifica caratteristica o sfumatura astratta del significato di quel token, decisa autonomamente dal modello durante l'addestramento.
Anche se noi umani non possiamo leggere direttamente cosa significa ogni singola coordinata, idealmente potrebbero rappresentare concetti come:


• È un oggetto animato o inanimato?


• Ha a che fare con il tempo (passato/presente)?


• È una parola formale o informale?


• Ha una connotazione positiva o negativa?


Mettendo insieme centinaia o migliaia di queste sfumature (numeri), il modello ottiene un'istantanea precisissima del significato del token. Quindi per noi non hanno alcun significato, non sappiamo tradurre il significato di quei numeri, perchè il modello li ha "autoregolati" durante il suo training, la fase in cui il modello impara. 

Se il nostro prompt è composto da N caratteri (o token) e abbiamo scelto una dimensione di spazio nascosto(cioè quanti di quei numeri decimali scegliamo) pari a n_embd, la nostra frase diventa a tutti gli effetti una matrice di ingresso X con dimensioni:


$$X\rarr N \times n\_embd$$


Questa matrice X contiene tutta l'informazione iniziale delle nostre parole, ma è ancora statica: i vettori non si sono ancora parlati tra di loro. Per iniziare a calcolare il contesto, la matrice X viene clonata in tre copie identiche e sparata in parallelo contro le tre famose matrici fisse del blocco di Attention: $W_{Q},W_{K}, W_{V} $. Tutte e tre queste matrici sono griglie quadrate grandi [n_embd × n_embd].

Attraverso tre moltiplicazioni matriciali separate, otteniamo tre nuove matrici fondamentali, ognuna delle quali mantiene la forma $[N \times \text{n\_embd}]$.

1. $Q = X \times W_Q$ (La matrice delle Query)
2. $K = X \times W_K$ (La matrice delle Key)
3. $V = X \times W_V$ (La matrice delle Value)

A questo punto avviene il calcolo dell'Attention. Il computer prende la matrice Q e la moltiplica per la matrice delle Key trasposta $K^{T}$, che ha dimensioni $[\text{n\_embd} \times N]$. Il risultato di questo prodotto riga per colonna è una matrice quadrata che mappa le relazioni tra tutte le parole della frase, chiamata Attention Matrix:

$$ Attention \space Matrix = Softmax(\frac{Q \times K^T}{\sqrt{d_k} }) \rarr [N \times N]$$

I numeri all'interno di questa griglia vengono divisi per un fattore di scala e passati dentro una funzione Softmax riga per riga, trasformandoli in percentuali di affinità che sommano a 1. Questa matrice ci dice concretamente quanto ogni parola debba pesare e "considerare" le altre all'interno del contesto.

Subito dopo, questa matrice di voti $N \times N$ viene moltiplicata per la matrice V. Il risultato di questo prodotto ci riporta  alle dimensioni di partenza:


$\text{Output Attention}=\text{Attention Matrix}\times V\rightarrow [N\times \text{n\_embd}]$


In questa nuova matrice, l'identità di ogni parola è stata "contaminata" dal significato delle parole circostanti.

I dati non si fermano qui. Questa matrice $[N \times \text{n\_embd}]$ entra dentro il blocco Feed-Forward Network (FFN) per subire un'elaborazione logica profonda. Qui incontra altre due matrici in serie che fanno un gioco di "imbuto" geometrico: la prima matrice spinge e allarga lo spazio vettoriale (di solito quattro volte più grande, es. da 512 a 2048), i numeri subiscono una distorsione non-lineare attraverso la funzione di attivazione GELU, e la seconda matrice schiaccia nuovamente il vettore riportandolo alla dimensione originale n_embd. Quindi nella prima operazione avremo un tensore allargato, 4 volte più grande, poi la seconda moltiplicazione lo riporta alle dimensioni originali.


Siamo alla fine del viaggio (fase di Inferenza). Al modello ora interessa solo prevedere il token successivo, quindi effettua un'operazione di slicing ed estrae esclusivamente l'ultima riga della matrice, ovvero il vettore lungo n_embd associato all'ultimo carattere del prompt, che ormai ha accumulato lo storico e il significato di tutta la frase.
Questo singolo vettore riga viene "sparato" contro **l'ultimissima matrice della rete, la Language Model Head (LM Head)**, che ha dimensioni 
[n_embd × dictionary_size].

La moltiplicazione genera quindi un vettore riga finale lungo quanto tutto il nostro dizionario (es. 256 indici per i caratteri UTF-8):

$\text{Logits}=\text{Vettore\ Ultima\ Parola}\times \text{LM\ Head}\rightarrow [1\times \text{dictionary\_size}]
$


I 256 numeri ottenuti passano per un'ultima Softmax che assegna a ogni carattere una probabilità. Il computer prende l'indice con il valore più alto (o ne campiona uno in base alle percentuali), scopre a quale carattere/parola corrisponde e tira fuori l'output.

Per generare l'intera risposta, il modello funziona in modo autoregressivo: prende l'ultimo token che ha appena predetto, lo attacca alla fine della frase iniziale e fa rientrare tutto il blocco da capo nella rete. Questo ciclo si ripete continuamente, un token alla volta, finché il Transformer non decide di fermarsi o non raggiunge la lunghezza massima stabilita.

**TRAINING**

Tutto il procedimento descritto sopra funziona perfettamente solo se le matrici della rete $W_{Q}, W_{K}, W_{V}$, le matrici dell'FFN e la LM Head) contengono i numeri "giusti". Ma all'inizio del progetto, nel Giorno 0, tutte queste matrici sono riempite di numeri microscopici messi completamente a caso. Se inseriamo un prompt, il modello sputerà fuori caratteri casuali e senza senso.
La fase di Training (Addestramento) serve proprio a sintonizzare questi milioni di numeri per insegnare al Transformer la struttura della lingua. Si tratta di un gigantesco ciclo iterativo programmato per girare un numero fisso di volte (max_iters).


1. Il **Forward Pass e il calcolo della Loss**:
Il computer non estrae una sola frase dal nostro dataset di testo, ma raccoglie un gruppo di frasi insieme (il batch_size) tagliate a una lunghezza massima fissa (il block_size). Questo blocco di dati viene sparato nella rete. Il Transformer esegue tutte le moltiplicazioni matriciali viste nella pipeline e sputa fuori le sue scommesse per ogni posizione della frase.
Immediatamente dopo, il codice confronta la scommessa fatta dal modello con il carattere reale che si trova subito dopo nel nostro testo di addestramento. Da questo confronto viene calcolato un singolo numero che rappresenta la mazzata di errore commessa dalla rete: la Loss Function (funzione di errore, solitamente calcolata tramite la Cross-Entropy Loss). Se il modello indovina la Loss è vicina a zero; più la scommessa è stupida e lontana dal bersaglio, più la Loss sale.

2. **La Backpropagation (Il viaggio di ritorno)**: una volta ottenuto il valore numerico della Loss, il computer blocca il flusso e inverte la marcia, avviando la Backpropagation (propagazione all'indietro dell'errore).
Partendo dall'ultimo strato e risalendo verso l'inizio della rete, l'algoritmo applica la regola della catena (Chain Rule) dell'analisi matematica. Per ogni singola matrice fissa, calcola la derivata parziale della Loss rispetto a ogni singolo peso w contenuto nella griglia
Questo calcolo risponde a una domanda chirurgica: "Se io modifico di un millimetro questo specifico numero w dentro questa matrice, la Loss totale salirà o scenderà?". L'insieme di tutte queste derivate parziali prime calcolate per l'intera matrice forma il vettore Gradiente. Il gradiente è la bussola del modello: indica la direzione della pendenza dell'errore e quanto quel singolo peso sia "colpevole" dello sbaglio finale.

3. **L'Aggiornamento dei Pesi (Optimization)**: trovata la direzione grazie al gradiente, entra in gioco l'ottimizzatore (es. AdamW). Per evitare che la rete faccia salti troppo violenti che distruggerebbero l'equilibrio dei calcoli, interviene il learning_rate, che agisce come una vera e propria "valvola" per frenare la correzione.Se il gradiente è positivo significa che il valore del peso stava facendo aumentare l'errore, quindi il segno meno lo riduce. Se il gradiente è negativo, meno per meno fa più, e il valore del peso aumenta.
# Tokenizzazione 
 Essendo un progetto didattico non uso la tokenizzazione BPE(Byte Pair Encoding), ma uso  semplicemente la tokenizzazione a livello di carattere. La BPE è molto efficiente perchè è un ibrido dell'approccio per parola e per carattere: cerca nei testi di addestramento  che generano il vocabolario le coppie di byte o caratteri più frequenti, ad esempio prima "es", poi vede che "est" è più frequente e unisce i pezzi: il token generato nel dizionario sarà associato a quel gruppo di lettere, es. token:756. Nella tokenizzazione per parola quando viene passata una parola sconosciuta, il tokenizzatore può andare in tilt o associare un nuovo token fisso (come < UNK >) per ogni parola sconosciuta, perdendo informazione quindi. Invece il problema della tokenizzazione per carattere è che si formano matrici con troppe righe (sequenze lunghe): usando UTF-8, avremo un vocabolario base di 256 ID, ma siccome ogni singolo carattere è un token indipendente, una frase genererà una matrice con un numero immenso di righe...

# Iperparametri 

## ```batch_size = 32```

Il batch  è semplicemente un gruppo di frasi che la rete neurale elabora contemporaneamente durante il training(nella realtà viene usato anche nell'inferenza, cioè quando il modello risponde ai prompt, cosi che un modello può rispondere a più utenti contemporaneamente), tutto insieme, in un colpo solo.
Invece di dare in pasto al tuo Transformer una frase alla volta, aspetti di averne raccolto un gruppetto (ad esempio 8, 16 o 32 frasi) e le "spari" dentro le matrici nello stesso millisecondo. Inoltre il batch viene usato per calcolare l'errore: nel training, quando viene calcolato l'errore per aggiornare i pesi, l'errore viene calcolato sul batch e non su una singola frase: se correggi i pesi delle matrici ($W_Q, W_K, W_V$) basandoti su una sola frase, rischi che quella frase sia un'eccezione strana e che i pesi cambino nella direzione sbagliata. Se invece calcoli l'errore medio di un batch di 16 frasi, la direzione del gradiente sarà molto più stabile e pulita, perché gli errori delle singole frasi si compensano tra loro.

## ```block_size = 64```

All'interno del singolo batch, ogni riga contiene esattamente 64 token consecutivi. All'interno di una sequenza da 64, il modello impara contemporaneamente 63 predizioni: predire il 2° carattere dato il 1°, il 3° dati i primi 2, fino a predire il 64° dati i primi 63. In pratica è la lunghezza massima della frase che il tuo modello può ricevere in ingresso. Se ad esempio diamo una frase di 100 token, il nostro modello scarta tutto ciò che va dal 65 token in poi. Quindi per elaborare il resto dovremo dare un altro prompt...

## ``` max_iters = 1500```

Il max_iters (o maximum iterations) è  un contatore che indica alla rete neurale quante volte in totale deve ripetere il ciclo di addestramento prima di fermarsi.Impostare max_iters = 5000, ad esempio, significa dire al computer: "Prendi un batch di dati, calcola l'errore, aggiorna le matrici Q, K, V e ripeti questa operazione esattamente 5.000 volte. Alla 5001-esima, fermati e salva il modello". Quindi per 5000 volte viene ricalcolato l'errore e aggiornati i pesi in poche parole. Non confondere però il concetto di epoca:una epoca significa far vedere al modello tutto il tuo dataset di testo intero, dall'inizio alla fine, esattamente una volta.

## ```eval_interval = 300```

un parametro che imposta ogni quante iterazioni il computer deve "fermarsi" per calcolare l'errore del modello su un certo numero di batch(eval_iters) sia del train set che del test set e mostrarlo a schermo.  Questo controllo serve a monitorare l'eventuale insorgenza dell'overfitting (quando la loss di train continua a scendere, ma la loss di validazione inizia a salire).

## ```learning_rate = 1e-3 (0.001)```
Il learning rate (chiamato anche tasso di apprendimento) è la grandezza del passo che il modello fa quando corregge i numeri delle sue matrici fisse dopo aver calcolato l'errore con la backpropagation. In altre parole indica quanto velocemente (e quanto aggressivamente) il tuo Transformer deve imparare dai suoi errori.In particolare l'aggiornamento di ogni peso nella matrice, durante la fase di training avviene in questo modo:


$$w_{nuovo}=w_{vecchio}−(Learning Rate×Gradiente)$$

w è infatti il singolo elemento delle matrici dei pesi. Il learing_rate è come una "valvola" che regola quanto questi pesi vanno aggiornati.Per Gradiente intendiamo nello specifico il gradiente della funzione di errore (la Loss) rispetto a quel singolo peso w:

$$ Gradiente =  \frac{\partial Loss}{\partial \omega} $$

In realtà per essere precisi più che gradiente dovremmo chiamarla derivata parziale, perchè in analisi matematica il gradiente è un vettore di derivate parziali prime. 

## ```device = 'mps' if ... ```
Serve a scegliere automaticamente l'hardware più potente disponibile sul computer per fare i calcoli(operazioni tra tensori, cioè matrici multidimensionali). Qui la scelta è tra mps(adatto per mac) e cpu... 
- cpu: Calcolo seriale generico eseguito sui core della CPU.   
- cuda: Piattaforma Nvidia per l'accelerazione su GPU dedicate.   
- mps (Metal Performance Shaders): Il framework di accelerazione grafica e di calcolo vettoriale di Apple integrato in macOS, che delega le operazioni sui tensori direttamente ai core della GPU dei processori Apple Silicon tramite la memoria unificata

## ```n_embd   ```
 è la lunghezza del vettore associato a una singola parola. Indica quante "caratteristiche numeriche" usi per descrivere il significato di quel token. Se metti n_embd = 512, significa che la parola "mela" sarà una lista di 512 numeri. Questa dimensione la decidi tu(es. 128, 512, 768) in base a quanta memoria ha la tua GPU e a quanto vuoi che il modello sia intelligente.

Questo parametro è molto importante perchè deciderà la dimensione delle matrici dei pesi. Le matrici $W_q,W_k,W_v$  . Infatti ciascuna di queste avrà dimensione $n\_embd \times n\_embd$. 



## ```n_head````
Quando progetti un Transformer, tu devi scegliere due numeri obbligatori per definire la dimensione della rete:
1. n_embd: quanti numeri totali compongono una parola (es. 64).
2. n_head: in quante parti vuoi dividere questi numeri (es. 4).
Da questi due numeri che hai scelto tu, il computer calcola automaticamente un terzo valore fisso, che nel codice si chiama head_size (la dimensione di una singola testa):
$$\text{head\_size}=\frac{\text{n\_embd}}{\text{n\_head}}=\frac{64}{4}=16$$
n_head definisce il numero di 'teste' di attenzione. Serve a dividere il vettore di embedding in più parti uguali, permettendo al modello di analizzare le relazioni tra le parole da più punti di vista contemporaneamente (ad esempio, analizzando la grammatica con una testa e il significato logico con un'altra) nello stesso istante. Questa suddivisione viene effettuata solo per il calcolo dell'attention. Infatti prima di entrare nel FFN viene ricomposta la tabella originale: il computer poi prende queste 4 matrici e le incolla una di fianco all'altra, in orizzontale, ricostruendo la tabella originaria.Solo dopo passerà nella FFN.

## ```n_layer````
L'n_layer (numero di livelli o strati) indica quante volte l'intera macchina che abbiamo appena descritto viene replicata e messa in fila, una dopo l'altra.
Se nel tuo progetto imposti n_layer = 6, significa che la tua frase non passa dentro un solo blocco di Attention e Feed-Forward, ma attraversa 6 blocchi identici messi in serie.
Ogni layer ha le sue matrici addestrate indipendentemente dagli altri layer. 

       [ Prompt Iniziale ]
                │
                ▼
      ┌───────────────────┐
      │     LAYER 1       │ ──► (Attention + FFN con i propri pesi)
      └───────────────────┘
                │ [N x 64]
                ▼
      ┌───────────────────┐
      │     LAYER 2       │ ──► (Attention + FFN con altri pesi)
      └───────────────────┘
                │ [N x 64]
                ▼
               ... (fino a n_layer)
                │ [N x 64]
                ▼
      ┌───────────────────┐
      │  LM HEAD (FINALE) │ ──► (Sceglie la parola dal dizionario)
      └───────────────────┘

La cosa fondamentale da capire è che questi blocchi (i Layer) sono disposti in verticale (in serie), come i piani di una fabbrica.
1. La tua frase entra nel Layer 1.
2. Dentro il Layer 1 fa tutto il giro che abbiamo studiato: entra nell'Attention (si divide in n_head, si calcolano Q, K, V, si ricompone) e poi entra nel Feed-Forward (allargamento e riduzione).
3. L'output che esce dal Layer 1 ha ancora dimensioni [N × n_embd] (es. [4 × 64]).
4. Questo output diventa direttamente l'input del Layer 2.
5. Il Layer 2 rifà esattamente le stesse identiche operazioni, ma usando le sue matrici di pesi personali, e passa il risultato al Layer 3.
Questo telefono senza fili matematico continua fino all'ultimo strato (n_layer). Solo quando i vettori escono dall'ultimissimo Layer, il codice estrae l'ultima riga e la manda alla LM Head per scegliere la parola dal dizionario.


## ``` dropout ```
l dropout è una tecnica di regolarizzazione che serve esclusivamente durante la fase di training per impedire al modello di imparare il testo a memoria (Overfitting).
In parole molto povere: ad ogni iterazione di addestramento, il computer "spegne" a caso una percentuale di neuroni, costringendo la rete a trovare strade alternative per indovinare la parola successiva.
Noi qui abbiamo impostato dropout = 0.1 (ovvero il 10%):
• Durante il Forward Pass, PyTorch prende le tue matrici o i tuoi vettori e, in modo del tutto casuale, sostituisce il 10% dei numeri presenti con uno zero assoluto (0.0).
• Al passo (iterazione) successivo, riattiva quei neuroni e ne spegne un altro 10% pescato a caso.
Senza Dropout la rete tende ad affidarsi sempre agli stessi identici accoppiamenti di numeri per memorizzare le frasi del dataset. Se provi a usarla fuori dall'addestramento, basta una lettera diversa per mandarla in crisi.


