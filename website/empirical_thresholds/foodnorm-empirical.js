// Weebly result-page script using empirical ZL/ZU by household size.
// Generated from the revised active/sedentary analysis dated 2026-09-28.
function renderResults() {
  let people;
  try {
    people = JSON.parse(localStorage.getItem('people'));
    if (!Array.isArray(people) || people.length === 0) throw new Error('Missing household');
    people = people.map(person => {
      const age = Number(person.age);
      if (!Number.isInteger(age) || age < 0 || !['male', 'female'].includes(person.sex)) {
        throw new Error('Invalid household member');
      }
      return {age, sex: person.sex};
    });
  } catch (error) {
    window.location.href = '/';
    return;
  }

  const results = getResults(people);
  const walker = document.createTreeWalker(document.body, 4); // SHOW_TEXT
  let node;
  while ((node = walker.nextNode())) {
    if (node.parentElement.closest('script, style, noscript, textarea')) continue;
    node.nodeValue = node.nodeValue.replace(
      /\[(foodNormActive|foodNormSed|basketsActive|basketsSed|lowActive|lowSed|highActive|highSed)\]/g,
      (match, key) => {
        const value = results[key];
        if (!Number.isFinite(value)) return 'לא זמין';
        return Math.round(value).toLocaleString('en-US') + (key.startsWith('baskets') ? '' : ' ₪');
      }
    );
  }
}

function getResults(people) {
  let sumActive = 0;
  let sumSed = 0;
  people.forEach(person => {
    if (person.age < 2) return;
    let groups = person.age <= monthlyCalorieIntake.child[0].age.max
      ? monthlyCalorieIntake.child : monthlyCalorieIntake[person.sex];
    const group = groups.find(item =>
      (!('min' in item.age) || item.age.min <= person.age) &&
      (!('max' in item.age) || item.age.max >= person.age)
    );
    if (!group) throw new Error(`No calorie group for age ${person.age}`);
    sumActive += group.intake.active;
    sumSed += group.intake.sed;
  });

  const basketsActive = sumActive / caloriesPerBasket;
  const basketsSed = sumSed / caloriesPerBasket;
  const foodNormActive = basketsActive * minBasketPrice;
  const foodNormSed = basketsSed * minBasketPrice;
  return {
    foodNormActive, foodNormSed, basketsActive, basketsSed,
    ...getEmpiricalThresholds(people.length)
  };
}

// Fixed monthly NIS thresholds calculated from the survey small samples.
// null means the prescribed sample contains no qualifying household for that size.
const empiricalThresholdsByHouseholdSize = {
  1: {lowActive: 1928.7857846562, highActive: 1459.3179169577, lowSed: 1226.2700000000, highSed: 1787.3300000000},
  2: {lowActive: 2731.1555045207, highActive: 5072.3386462713, lowSed: 2861.5900000000, highSed: 2929.5996062992},
  3: {lowActive: 5330.1898300692, highActive: 7141.0299469035, lowSed: 4082.3600000000, highSed: 7771.8565000000},
  4: {lowActive: 6793.9814511559, highActive: 6459.7687972100, lowSed: 4731.4800000000, highSed: 5909.5782500000},
  5: {lowActive: 6415.2330301365, highActive: 10129.5906362062, lowSed: null, highSed: 7057.6396629213},
  6: {lowActive: 8412.8424163745, highActive: 14968.5185024100, lowSed: 6961.0850000000, highSed: 7100.0011764706},
  7: {lowActive: 11708.1163549108, highActive: 16304.5355111765, lowSed: null, highSed: 14465.1081818182},
  8: {lowActive: 12708.0497557354, highActive: 20636.2749509055, lowSed: 12738.3600000000, highSed: 11937.3966666667},
  9: {lowActive: 14441.8877217493, highActive: 22443.4891662493, lowSed: null, highSed: 14064.6600000000}
};

function getEmpiricalThresholds(householdSize) {
  const values = empiricalThresholdsByHouseholdSize[householdSize];
  if (!values) throw new Error(`No empirical thresholds for household size ${householdSize}`);
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, value === null ? NaN : value]));
}

const monthlyCalorieIntake = {
  child:[{age:{min:2,max:3},intake:{sed:30417,active:42583}}],
  female:[
    {age:{min:4,max:8},intake:{sed:36500,active:54750}},
    {age:{min:9,max:13},intake:{sed:48667,active:66917}},
    {age:{min:14,max:18},intake:{sed:54750,active:73000}},
    {age:{min:19,max:30},intake:{sed:60833,active:73000}},
    {age:{min:31,max:50},intake:{sed:54750,active:66917}},
    {age:{min:51},intake:{sed:48667,active:66917}}
  ],
  male:[
    {age:{min:4,max:8},intake:{sed:42583,active:60833}},
    {age:{min:9,max:13},intake:{sed:54750,active:79083}},
    {age:{min:14,max:18},intake:{sed:66917,active:97333}},
    {age:{min:19,max:30},intake:{sed:73000,active:91250}},
    {age:{min:31,max:50},intake:{sed:73000,active:91250}},
    {age:{min:51},intake:{sed:66917,active:85167}}
  ]
};
const caloriesPerBasket = 53096.57;
const minBasketPrice = 692;

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', renderResults);
  else renderResults();
}
