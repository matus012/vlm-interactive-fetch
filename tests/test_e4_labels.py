"""E4 C1 labels (task 014): switch off = byte-identical worlds; switch on = one category per box, deterministic."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sim.constants import BOX_COLOURS
from sim.world import generate as gen

REPO = Path(__file__).resolve().parents[1]
# sha256 of generate(seed, blocker).xml, seeds 1000-1019 x none/crate/cabinet, per house, computed with the code
# BEFORE the LABELS switch was added (task 014).
REF = """A 1000 none dad6be57663d2380661b6356d0c97d27739b5d4a9b02fd26f2239df689fc83bd
A 1000 crate 73c3293eef23ef2072016c0a15decc3ff902863403244621b4975703a0c92317
A 1000 cabinet 55a38f8de26122b0c345078f57fe7222f07b10450782f8eb85debfa0d74b3d4b
A 1001 none e7130337a2dff443f4d262d549a6cd6d1fe85fa29b999ad15068442d369bbcfd
A 1001 crate 072f3e6442153999048aa8c229054176020b40aea83540fffe6cab6a45f4fa29
A 1001 cabinet 47f3b50bdcd951acef9ac94b8dd8252f0919967427d73d696d55cd89b78adcc0
A 1002 none 1e46352d8f717fbe31c51fa045ecac2d13c7c3a5ad410bae82b19d2b50555ccd
A 1002 crate 582cc0a751d9eb338425e54d0d9daf9d3fa24c56188752ca3a50b6c0cafc0a09
A 1002 cabinet f78be9a5e3b0457345a56a6cbbeb82fe7152e865a6b996f2df72aee121749d77
A 1003 none d90d4d90b2469676dac87508354665dd6bc334183803bb6401c6ad3892df5cb5
A 1003 crate fcda1b64c1f411069741afefff523b2a320b70f780342bfe8484295123cdf846
A 1003 cabinet 954204658b5adbbe80ed38baf662829ae5e97cf7bc35a20c6de3ffa93728122c
A 1004 none 46405ffb1c53f25deae3554fc3112f43622e5c4fb558a3276005b4fc65d4fff9
A 1004 crate 6bc48a03f1fcb68817451d4f3e7d2c37f56e6d0ac5450a27c7cf1363ea497b6b
A 1004 cabinet 9329a4c1c19ab699a66ae1a1567c012a40b5c426438cb4c051ad72234b4afe8b
A 1005 none 135a3dbf32c3928582bbeb51e2f2c599de98372d357e6137bac52c1104bda833
A 1005 crate 2333d8c7f5a2e2df4ae041184f8c38551396b3b51f792ecb9bf0a64d5eddb49d
A 1005 cabinet e8dbbfec24bc05fc11bf62f773b6765413ed9c3fa3eeda3df249b6a2789eefcd
A 1006 none e773768d0e1fcbf30e4cab29b44f5125529db83e5e90d54a80a16773e45fba3f
A 1006 crate db24a87398b78f985f748742b0922f29e537f71250381123460b1da2581b2fbe
A 1006 cabinet 6ebf168900b790d5f78068ee45d1508d9db734a47a56dfc31435a3e47927f4db
A 1007 none 1d2806999433fd6af4c5092db3007904746bed2889986e40f919ce87b9c381bc
A 1007 crate 1c897d95e545d63f1618e4dbd69b42233dc55f0f0113f6998a4cefde7a2f3b61
A 1007 cabinet 990e1f6234b34bc8630d12376ad3966bde70b1ddb6f18a6f922bfe67a05ff574
A 1008 none 3bea3c68da608311dde8d26c9dd6e32cf00c71829173b6a76f00d55f88d9dd78
A 1008 crate ee796e188cd66e8410dff61c9936dee0c2d507529062fd4ff69e6eec92b0734a
A 1008 cabinet 2c10b4fa1ddece8026e6afc2715273db5079c52a238304ebb8b99e4fba458bcc
A 1009 none 6bc8bed122aaaed8539a7bee0eedb830341446919c9eec777c27122bccf31e8d
A 1009 crate 8f62f87375c94dc5c5fb3ca4ce02441059744ef963cddb8c0c93963acc3597f5
A 1009 cabinet a74bd5dafd598566e0afa5fc2adce268149bccab866813e1db256dd000ec79cd
A 1010 none 5960f626bd5d9a3b62cadafbc487e33c21cf5df737f267f1723470aea404257d
A 1010 crate 714a6eec52d143441de22631b352b01f1904741ed601d8bf20d39f99e4be5d1b
A 1010 cabinet cd0002325224cc55d98370ac9f43d472621ec7b3603035a18cd316c87c3124b1
A 1011 none 06929dfb096454967f9450d5780832149e2f81326ae366265a814f2e6253c041
A 1011 crate 8f02f12dddc20447ce076843956799a73b27e4bb327b99891ff37e0c55971c5f
A 1011 cabinet f4b08b2d7099c504b0995b2e689af45405d48f65ba992028fde2955606c1337b
A 1012 none 7b8880df2ac4b0b559e4e5cb6329ca44413a428db9c32d9a0d21a132f607d731
A 1012 crate ef69505f3cb10859847fbb6b8a2e91697140ac22babe7df5bddcf91a57380413
A 1012 cabinet 421278fe7d4cc5ef3bbda37a100b55de77cc4b6d8fa1f1cd0cce39f1cef58dc5
A 1013 none 75197e0ff84ea12a18313a81aa25bc6eed67caf413539700fc2c518722899893
A 1013 crate 3a3fa18ff7e39149c454cc3a40c256c7abf6e7452de2f1f58473b8b31e79b690
A 1013 cabinet 228f495b7da2bbcec4bdd4df60b124ccfcdc38e58269a97c02e960f76b531f17
A 1014 none 10cfcb441e206606b005e6784cf9c139a8c85c076d1cc3420060b013a9642bea
A 1014 crate d9fe87adbf0263f0d6ecb00d04a0ae7ee66644649ac74a8af9c7ff3ddbc46d06
A 1014 cabinet 5b4db450a01419b6cd5da47a194f4ef2abeb84e1c3463e2e5ef35ebe33930b66
A 1015 none 27f4f426dc733e7f498c2e3d46125a728d73889bb54e5419303b2c0382bde49e
A 1015 crate 5df13f8e9e65f18784b4b30b46d6ea5026142e84cc88cab0aca0e5f9874762d7
A 1015 cabinet e119a3e043097b1e6deb90394d9f40c8a5ed2119e27d0e9aaf87227c180a1b4e
A 1016 none 2fbb2a4764211e1e0d32c2457efa84728d43562a10d107f2c55f104ec0d503bf
A 1016 crate 123ed0de57ede8ec35bcb8a73ae82aca48b24dad7e32ebcb78d8c7fd3d363966
A 1016 cabinet e8b1d2559812fcfd0ea586f1b4046168e7a53615fd88fb1e9a6063677f84601f
A 1017 none 0ead3ac3a3670b6bf71fec752c3ddf5a4dba675a6cdc97edd9af4546a30f4075
A 1017 crate 94909e26dc07583bd6de7ea4912c5d9a6b12ee75b7bca99852f494100f3476e6
A 1017 cabinet 7efb21c0d60f52dfea225025244d071c36b9d9decb3a1ceca32c660ea4da8452
A 1018 none acd2012e73e3f5981775348143c6e0f3ddcda503853ad98165d0777427f0e52d
A 1018 crate c586115d82dd2b7d6ca9a34d4342edf64753ee4778fe923d50868df03abd1e36
A 1018 cabinet 0510c19f03e60eb3342b2628a7295541ef76fcd3d2208a13a1fa887bd79579cc
A 1019 none e749661aca1666b67cf00baad39c4a364d52451216f9046deea0b1495a6c82fd
A 1019 crate 5000d3ae26d0d6bf8104b4f11e9ac9e1685029e030e994c0fc66d517ce699e99
A 1019 cabinet 840f0bd39d5e52dff4ad0a1b291ce63404e234fa916d402622739dbc41e388ae
B 1000 none 61a45c13090b944b57d6423dba0e635c9a028a8fd771c64208a513d000ee86c0
B 1000 crate 8ec26db1f90fccb6cc6c5a4fc1f2728dcdf74d82a88a9f7fc92119ddc1b4345e
B 1000 cabinet 22b54c7439a16dfb5635694e348ddd5882a8b037ca75cb4e21bc5b86c74c6b7a
B 1001 none 1ed2982865df0f59da95dcef32cddffc09a4db175adffca2462bd6a363f32662
B 1001 crate afb041e7df1ce305b5f857aeac5fb79d1ea6c3392904ee4b51844b503acec276
B 1001 cabinet de9b250fe009c51cbcd0761d5d6fa6abdc998319c8601dbd021577f07b5debc2
B 1002 none 2374e118b8da93d04f4279db712b70c7fa0ce03fba9a1a6b95b1abd53c38c4eb
B 1002 crate 02f2d3210b2114a59cd9b784098d576eefe3851b1995f7c2b4d2df776e6973f2
B 1002 cabinet 3cbf5972b50e9d746b27d54145bb07825782ccabbb21382931515c2391d36fb4
B 1003 none 9300b447e9cae5e0f0d183212e856735adfbd103d9fdd8ec943fe0fb1c987f98
B 1003 crate 9b8afdfd2aa53083dceefc941826916d5be6d564cf7851cc664375b61a145bcd
B 1003 cabinet fbd35acfd701bc14078b1297753cae0dfc4fadcbdf1028f63e0c4340200740aa
B 1004 none b9acdc0e5af0cf5bcbc925b5a7897d49e326c3f60d95ed467e09233306128b6b
B 1004 crate 87d6b8372d26e91b7f08be527c5910b134a983dfa5e797e071fa9fe0c636cf18
B 1004 cabinet 803128da4594fd09bd2d9ca3f7768ef89f9c0674a118e080ca26c8c9023d4652
B 1005 none 18a0674e2c7cd307648c11a996887e763038c41592619d21aec4a60472289f9e
B 1005 crate f01ed58df23454cdc47a29718305edb9489586c91477304784d318679fef53d3
B 1005 cabinet 719429232e5b492fa1b783d8a7f2a53f55398ee447baf5b783916a06dfc72620
B 1006 none d5c5e674a68f7448b9f0b1deabbdc007e5b381f90e951cb58d2c64b62bba4f0c
B 1006 crate d9ee4a3ac2e948ca0f99e5e3e127caa9a2a0783c288cf1dd0f3663a1280a89dd
B 1006 cabinet dd90609ce18f75303797fe5bcb5b6cfe6b0bdc8877d41784cc29e5964788a096
B 1007 none 216988da8884890a7bd4ae5059f3c0027b1661ef0f05793cdaf1315eb227be2d
B 1007 crate a3d2657778d291fdc7a840ac3af159f7a54eba822e86e3db9813cc438368c1aa
B 1007 cabinet a3ce824ac8f0236a7733138be08d76ab22f9970ae78a7fb53d615d0899825ea3
B 1008 none ee4ab45e6932836b6e64bd20fe4f87764d93d8d33e9b68564d2ad82f25265526
B 1008 crate 896e4039b5b4c9c40de73cbe8248e797648e74fb55d7913fc51dd2f0c7b04f12
B 1008 cabinet ccb900a8d1fcdf063cd3961eec2232b5c37d9760fc83c0556b3b8c18e673da48
B 1009 none 1ce46ff1a50971da932f5543c1871ad3218b51e034a6015af01c7e247123ce71
B 1009 crate 2444e74b0ba68c30858fca1ac62ee0baec80db106c1581cc32e34ac7d87c9a31
B 1009 cabinet fab5696fcd2496c4a6816c4e23d622b592f1edc01abfcd8d6804f4fc0452feef
B 1010 none 1d2102809a02f7065095771bba27580adc31731119e983b23606ca524d4cc1fc
B 1010 crate 45b747f009ae2148f8474b25e1bc649fcdf7b06e27da9c4bf845a16ec29562c5
B 1010 cabinet b20ab752a45e36b9cea34004f35d5b1406eee86ff675d325e2c63cb6fabbf234
B 1011 none 5f9d83c78c44134022adfd223a6cb3c3d2080273ba728a7ad203ec314dd0ba53
B 1011 crate ff3d5bb7ee49b2f7d1c6070407a2e51ccb0da4b1446a242dbc09879030daec50
B 1011 cabinet 435b696c14ba419a7d05104247975e4aa686df5a1f57acc27dcb980e55e4afdb
B 1012 none 2d37ab8e676b895b36f7e8084087a6334e1d161e8f1dc45d5e2af5a5b4d3e8bf
B 1012 crate 533151cd80d7e12d0e1606aa3aa03129e517c267c325441945afac2bf791e81c
B 1012 cabinet 91948128fda437575a5dbc0eebb7b62928fa23521e8740c1df39b13c5bdfd4e0
B 1013 none b24c833cefd4b072a12c87041e5e2d25e881d20be88c199a83599aceb5805bc9
B 1013 crate f4ff5a07b86a4e5dedc60db72275805e7ebfe84ef50d23a71188c38356ca8bd9
B 1013 cabinet e9b22f509b4117faf76daa804e9fb20d707c933c5b6dd46f8700047f5f05ba0f
B 1014 none 87e2ae07166ab659534028921c6b83635963bce93602d91823840edae7ab5472
B 1014 crate b95aa52e0f21b2b18b351bb446ee8cf0cc9a871d68c2d00087946b8b16caae13
B 1014 cabinet 8dc6188a9a4a5bf3d9de822acf45abd5e6f94e555469ea9196ce1cf7920799f8
B 1015 none 1d383d7284ea2281fade8886b08803e64ddcbb943823b277d1b9eb4fb30633ac
B 1015 crate 2578bef2fa7a8a8a6a6c0a3218d4c3d2aecb6260a6d61f2966ca0914aff4d7bd
B 1015 cabinet d8ab5b6ae0a1e5afb8f0bdba6b71dc9bd40e54192de72236ffd7e27793add540
B 1016 none 5dbbdf36ee6200ccdac672b31434826d626c518231545be4640aadc64d261f0c
B 1016 crate e4656d7a58829f055c8817b2beda79e006a6d53f4c727b2f39f4d74f2cf76b06
B 1016 cabinet c601f2d13f503b578e6093ffa740f18019f42f0960425c33dfd335001c7f6fd0
B 1017 none e6894f862f1675f3156666faaa15fb0666ba423173c3ed47c37279114e50711f
B 1017 crate 9b449193cc4bf3bdc4ff5933294e20a9ce2c2e0811c8cb8575b545cca4c96597
B 1017 cabinet 1b75cc3ce2e53670aeab3a16f925c31c28f543fa2695105760005e42b29259d2
B 1018 none ea508d3d61f014de1263c9461d32d2f229d6fe1999988020bb4b816fee4589b0
B 1018 crate 5441119683121abc3336c67dc50aa2ed83812f546878422ee73e4f3dcc633644
B 1018 cabinet c88e31b2fc2b1396672bbd25c874d3e8c9d75851d7f9a93178392e7334268887
B 1019 none 664cb7f67309ee5f6b25dc0e60f2b4f58ad1cf75400e5e5d501d661ce28a5b77
B 1019 crate 3fed606c54d53d2e54c95391a46464bbde24371b3874e1f58e3de0bce866cc04
B 1019 cabinet 851d373fcf1ee008f1f472b0cd3ec5b5fa2c86408d44b3a10c0d15bcc4257bbc
"""
HASHES = """import hashlib
from sim.world.generate import HOUSE, LABELS, generate
assert LABELS is False
for s in range(1000, 1020):
    for b in ("none", "crate", "cabinet"):
        print(HOUSE, s, b, hashlib.sha256(generate(seed=s, blocker=b).xml.encode()).hexdigest())
"""


@pytest.mark.parametrize("house", ("A", "B"))
def test_switch_off_byte_identical(house: str) -> None:
    env = {**os.environ, "NAV116_HOUSE": house, "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", HASHES], env=env, cwd=REPO, capture_output=True, text=True,
                         check=True).stdout
    want = [ln for ln in REF.splitlines() if ln.startswith(house + " ")]
    assert len(want) == 60 and out.splitlines() == want


@pytest.fixture
def labels_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen, "_LABELS_DRAWN", set())
    monkeypatch.setattr(gen, "LABELS", False)
    gen.set_labels(True)


@pytest.mark.parametrize("seed", range(1000, 1010))
def test_switch_on_one_category_per_box(labels_on: None, seed: int) -> None:
    w = gen.generate(seed=seed)
    labels = [b["label"] for b in w.boxes]
    assert [b["colour"] for b in w.boxes] == list(BOX_COLOURS)
    assert sorted(lb["category"] for lb in labels) == sorted(gen.LABEL_WORDS)
    assert all(lb["word"] in gen.LABEL_WORDS[lb["category"]] for lb in labels)
    w2 = gen.generate(seed=seed)
    assert w2.boxes == w.boxes and w2.xml.encode() == w.xml.encode()
    for b in w.boxes:
        assert f'material="label_{b["colour"]}"' in w.xml
        assert (gen.LABEL_DIR / f'{b["label"]["word"]}.png').is_file()


def test_switch_on_varies_and_keeps_poses(labels_on: None) -> None:
    ws = [gen.generate(seed=s) for s in range(1000, 1020)]
    assert len({tuple(b["label"]["word"] for b in w.boxes) for w in ws}) > 1
    for w in ws[:3]:  # the label draw is its own rng stream: box poses are those of the switch-off world
        gen.LABELS = False
        off = gen._draw_boxes(w.layout, w.seed, gen.load_places())
        gen.LABELS = True
        assert [{k: v for k, v in b.items() if k != "label"} for b in w.boxes] == off


def test_switch_on_compiles(labels_on: None) -> None:
    from sim.world.house import load_house

    m, _ = load_house(gen.generate(seed=1000))
    assert m.ntex >= 4 and all(m.geom_matid[m.geom(f"box_{c}").id] >= 0 for c in BOX_COLOURS)


def test_set_labels_refuses_after_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen, "_LABELS_DRAWN", set())
    monkeypatch.setattr(gen, "LABELS", False)
    gen.generate(seed=1000)
    gen.set_labels(False)
    with pytest.raises(RuntimeError):
        gen.set_labels(True)
    monkeypatch.setattr(gen, "_LABELS_DRAWN", set())
    gen.set_labels(True)
    gen.generate(seed=1000)
    with pytest.raises(RuntimeError):
        gen.set_labels(False)
