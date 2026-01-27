class PoissonSettings {
private:
    PoissonSettings() = default; // no public constructor
    ~PoissonSettings() = default; // no public destructor
    inline static PoissonSettings* instance = nullptr; // declaration class variable
    // SETTINGS: ==============================================================
    double EPS_X = 1.0;
    double EPS_Y = 1e-10;
public:
    // defines a class operation that lets clients access its unique instance.
    static double get_EPS_X() {
        if (!instance) {
            instance = new PoissonSettings();
        }
        return instance->EPS_X;
    }
    static double get_EPS_Y() {
        if (!instance) {
            instance = new PoissonSettings();
        }
        return instance->EPS_Y;
    }
};