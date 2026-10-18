# LLMs per tutti
Lo scopo di questo Readme è far capire informalmente a chiunque il funzionamento di base del progetto didattico, spiegando il codice che ho scritto nel main.py e i concetti alla base. 

# Introduzione - Breve Overview sugli LLM
Cosa accade da quando scriviamo un prompt a quando riceviamo il nostro output del modello? Inizia tutto dalla tokenizzazione: il nostro prompt è una stringa che viene convertita in una sequenza di token numerici (ID interi). Più avanti spieghiamo meglio questo procedimento, ora diamo una introduzione generica.
Quindi, questa sequenza di token viene trasformata in una matrice di vettori densi attraverso lo strato di Embedding. Se il nostro prompt è composto da N caratteri (o token) e abbiamo scelto una dimensione di spazio nascosto pari a n_embd, la nostra frase diventa a tutti gli effetti una matrice di ingresso X con dimensioni:

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



