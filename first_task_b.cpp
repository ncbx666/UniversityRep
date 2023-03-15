int main() {
	int numbers[26] = {};
	ifstream f;
	f.open("idiot.txt");
	while (!f.eof()) {
		string t = "";
		getline(f, t);
		for (int i = 0; i < t.size(); i++) {
			if ((int)(t[i]) <= 122 and (int)t[i] >= 97) {
				numbers[(int)(t[i]) % 97] += 1;
			}
			else if ((int)(t[i]) >= 65 and (int)(t[i]) <= 90) {
				numbers[(int)(t[i]) % 65] += 1;
			}
		}
	}
	f.close();
	for (int i = 0; i < 26; i++) {
		cout << numbers[i] << endl;
	}
	return 0;
}
