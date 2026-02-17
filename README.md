# Hallo :)
### Wenn ihr das Projekt korrekt kompiliert und laufen gelassen habt und wenn ihr nichts an den Einstellungen geändert habt, dann sollte der Output folgendermaßen aussehen:
- Zuerst kommen BASIC SOLVER TESTS auf einem kleinen 4x4 Gitter (```problem_size = 4```), die immer ablaufen zur Sicherheit, um zu überprüfen, dass alle Löser funktionieren und dieselbe Lösung bekommen (verglichen mit einer LU-Zerlegung).
- Danach laufen pro ```problem_size``` (Gittergrößen: $[2^2\times2^2]$ bis $[2^{11}\times2^{11}]$) Löser-Benchmarks für LU (nur, wenn Problem klein genug ist) und fürs Multigrid-Verfahren mit Informationen über einzelne Iterationen und Laufzeit bis zur Konvergenz (Defektnorm kleiner als $1e^{-5}$).
  - ![2048x2048 Multigrid Benchmark Output](/img/example_solver_benchmark_2048x2048.png "2048x2048 Multigrid Benchmark Output")
- Nachdem alle Löser für alle ```problem_size``` gebechmarkt wurden, dann findet am Ende eine Laufzeitanalyse statt mit approximierter Komplexität:
  - ![Complexity Analysis Output](/img/example_complexity_analysis.png "Complexity Analysis Output")

### Erklärungen zu den Einstellungen sind weiter unten.

# Um mein Projekt laufen zu lassen:
### Erst setup mit (muss nur ein Mal gemacht werden, wenn nichts an CMakeLists.txt verändert wird - sollte nicht notwendig sein): 
- Sicherstellen, dass man im *root* Ordner sich befindet, mit z.B.:
  - >cd root;
  - >cd "C:\Users\user\Download\bartek\root";
- Den build-Ordner in mit einem *Generator* erstellen und konfigurieren mit:
  - >cmake -G "*Generator*" -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -S . -B *build_Ordner_Pfad*
    - *build_Ordner_Pfad* ist beliebig aber ich empfehle einfach *../build* zu benutzen.
    - Als "*Generator*" habe ich persönlich "MinGW Makefiles" benutzt, falls es mit anderen nicht laufen sollte aus irgendeinem Grund.
    - -DUSE_OPENMP=OFF wird einfach benutzt, weil ich nicht wirklich es geschafft habe den Code mit OpenMP gut zu optimieren.

### Danach zum kompilieren folgendes benutzen (jedes Mal wenn man was in einer .cpp oder .h Datei ändert):
- >cmake --build *build_Ordner_Pfad*
- Wenn du 1:1 der Anleitung gefolgt bist, dann solltest du dich noch im *root* Ordner befinden und die Command lautet dann:
  - >cmake --build ../build


# BASIC EINSTELLUNGEN:
- in main.cpp (Zeilen 709 bis 729)
  - ```bVerbose``` stellt ein, ob Informationen (Reduktionsrate, Defekt und Reduktion) für jede Iterationen geprinted werden sollten.
  - die boolschen Variablen wie z.B.   ```bMultigrid``` oder ```bJacobi``` stellen ein welche Löser angewandt werden.
  - ```problem_size``` bestimmt die Größe des Problems in Bezug auf das Gitter, so dass für ein ```[problem_size X problem_size]``` Gitter ein Poissonproblem mit einer ```[problem_size^2 X problem_size^2]``` Poisson-Systemmatrix generiert wird.
    - Gerade ist es so eingestellt, dass eine for-Loop über ```problem_size``` als 2er-Potenzen $2^i$ iteriert von $2^2$ bis $2^{11}$.  
    Ich empfehle jede Änderung in Form dieser for-Loop zu machen, da sich darin auch Code befindet, der notwendig für die Approximative-Komplexitätsanalyse (Laufzeitanalyse) ist.


Die Zeilen main.cpp:709-729 für reference:

    // Benchmark Settings
    bool bVerbose = 1;
    bool bJacobi = 0;
    bool bGaussSeidel = 0;
    bool bMultigrid = 1; 
    bool bILU = 0;

    // Collect benchmark results for all solvers across all problem sizes
    SolverResultsMap all_results;
    for (int i = 2; i <= 11; ++i) {
        size_t problem_size = 1 << i;
        SolverResultsMap batch_results = run_solver_benchmarks(problem_size, 
            bVerbose, bJacobi, bGaussSeidel, bMultigrid, bILU);
        
        // Merge batch results into all_results
        for (auto& [solver_name, results] : batch_results) {
            for (auto& result : results) {
                all_results[solver_name].push_back(result);
            }
        }
    }

# ADVANCED EINSTELLUNGEN:
- in main.cpp bei jedem ```.set_convergence_params(...)``` kann man zu jedem Löser die Konvergenzparameter einstellen (ist dokumentiert).
  - z.B. in main.cpp:566 kann man das fürs Benchmark des Mehrgitter-Verfahrens machen.
    - Zusätzlich kann man da mit ```.set_params(...)``` in main:cpp:564 das Mehrgitter-Verfahren konfigurieren.

- in problems/headers/2d_possion_settings.h kann man mit ```EPS_X``` und ```EPS_Y``` ein anisotropes 2D-Poisson-Problem einstellen mit verschiedenen Kopplungsstärken je nach Richtung. Ist momentan so eingestellt, dass beide Variablen gleich 1.0 sind und somit das Problem isotrop ist.

- Das meiste ist so dokumentiert, dass man eigentlich durch main.cpp alle Einstellungsmöglichkeiten der Löser und der Benchmarks einsehen kann.