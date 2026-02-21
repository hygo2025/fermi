
| ID            | Ip    | Status   | Alg      |
|---------------|-------|----------|----------|
| 01 Janela     | 17    | Ocupado  | BERT4Rec |
| 02 Cafe       | 16    | Ocupado  | GRU4Rec  |
| 04 Mesa Hygo  | 54    | Ocupado  | BPR      |
| 05 Mesa Luiz  | 168   | Ocupado  | Pop      |


| 03-05                                                                    | Ip  | Status   | Baixado |
|--------------------------------------------------------------------------|-----|----------|---------|
| nohup bash -c 'MODELS="BPR" make benchmark' > benchmark.log 2>&1 &       | 168 | Sucesso  | Não     | 168
| nohup bash -c 'MODELS="ItemKNN" make benchmark' > benchmark.log 2>&1 &   | 16  | Sucesso  | Não     | 16
| nohup bash -c 'MODELS="Pop" make benchmark' > benchmark.log 2>&1 &       | 54  | Sucesso  | Não     | 54
| nohup bash -c 'MODELS="Random" make benchmark' > benchmark.log 2>&1 &    | 54  | Sucesso  | Não     | 54
| nohup bash -c 'MODELS="FOSSIL" make benchmark' > benchmark.log 2>&1 &    | 54  | Sucesso  | Não     | 54
| nohup bash -c 'MODELS="FPMC" make benchmark' > benchmark.log 2>&1 &      | 54  | Sucesso  | Não     | 54
| nohup bash -c 'MODELS="TransRec" make benchmark' > benchmark.log 2>&1 &  | 168 | Sucesso  | Não     |
| nohup bash -c 'MODELS="GRU4Rec" make benchmark' > benchmark.log 2>&1 &   | 54  | Sucesso  | Não     | 54
| nohup bash -c 'MODELS="NARM" make benchmark' > benchmark.log 2>&1 &      | 54  | Sucesso  | Não     | 54
| nohup bash -c 'MODELS="LightGCN" make benchmark' > benchmark.log 2>&1 &  | 17  | Rodando  | Não     | 17
| nohup bash -c 'MODELS="SASRec" make benchmark' > benchmark.log 2>&1 &    | 17  | Sucesso  | Não     | 17
| nohup bash -c 'MODELS="NextItNet" make benchmark' > benchmark.log 2>&1 & | 16  | Sucesso  | Não     | 16
| nohup bash -c 'MODELS="BERT4Rec" make benchmark' > benchmark.log 2>&1 &  | 17  | Sucesso  | Não     | 17
| nohup bash -c 'MODELS="Caser" make benchmark' > benchmark.log 2>&1 &     | 54  | Sucesso  | Não     | 54
| nohup bash -c 'MODELS="STAMP" make benchmark' > benchmark.log 2>&1 &     | 54  | Sucesso  | Não     | 54

Random Pop BPR ItemKNN FOSSIL FPMC TransRec GRU4Rec NARM LightGCN SASRec NextItNet BERT4Rec Caser STAMP

nohup bash -c 'POLL_SECONDS=60 MODELS="Random Pop BPR ItemKNN FOSSIL FPMC TransRec GRU4Rec NARM LightGCN SASRec NextItNet BERT4Rec Caser STAMP" make benchmark-queue' > benchmark.log 2>&1 &


